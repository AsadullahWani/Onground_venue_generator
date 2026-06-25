"""
FastAPI server for the Geoapify sports venue scraper.

Endpoints:
    GET  /health                         — liveness check
    GET  /sports                         — list available sport categories
    GET  /venues?city=...&sports=...     — search venues (sync, small queries)
    POST /venues/search                  — search venues (full options, body)
    GET  /venues/{place_id}              — single venue by Geoapify place_id
    GET  /geocode?city=...               — resolve city → place_id + lat/lon

Run:
    pip install fastapi uvicorn
    uvicorn api:app --reload --port 8000

Env vars:
    GEOAPIFY_API_KEY   — required
    GROQ_API_KEY       — optional, only needed for AI summaries
"""

from __future__ import annotations

import os
from typing import Annotated

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Import everything from the scraper module
from geoapify_sports_venues import (
    DEFAULT_SPORTS,
    SPORT_CATEGORY_MAP,
    CityGeocode,
    Venue,
    add_ai_summaries,
    geocode_city,
    scrape_geoapify_venues,
)

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Sports Venue API",
    description="Search sports venues via Geoapify Places API, with optional AI summaries.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # tighten this in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Pydantic response models
# ---------------------------------------------------------------------------

class VenueOut(BaseModel):
    place_id: str
    name: str
    city: str
    formatted_address: str
    short_address: str
    latitude: float | None
    longitude: float | None
    categories: str
    sport_type: str
    search_category: str
    website: str
    phone: str
    opening_hours: str
    country: str
    state: str
    postcode: str
    street: str
    distance: float | None
    ai_summary: str
    ai_sport_type: str          # AI-inferred sport(s), e.g. "Cricket" or "Football, Cricket"
    ai_confidence: str          # "high" | "medium" | "low" | "skipped" | ""

    @classmethod
    def from_venue(cls, v: Venue) -> "VenueOut":
        return cls(**{k: getattr(v, k) for k in cls.model_fields})


class VenueListResponse(BaseModel):
    city: str
    total: int
    venues: list[VenueOut]


class CityGeocodeOut(BaseModel):
    city: str
    place_id: str | None
    latitude: float | None
    longitude: float | None


class SportCategory(BaseModel):
    name: str
    geoapify_category: str


class SportListResponse(BaseModel):
    total: int
    sports: list[SportCategory]


# ---------------------------------------------------------------------------
# Request body model for POST /venues/search
# ---------------------------------------------------------------------------

class VenueSearchRequest(BaseModel):
    city: str = Field(..., description="City name, e.g. 'Kolkata' or 'New Delhi'.")
    sports: list[str] = Field(
        default_factory=lambda: DEFAULT_SPORTS,
        description=f"Sport names to search. Available: {', '.join(SPORT_CATEGORY_MAP.keys())}",
    )
    max_results: int = Field(default=50, ge=1, le=500, description="Max venues to return.")
    page_size: int = Field(default=20, ge=1, le=500, description="Results per Geoapify request.")
    include_ai_summary: bool = Field(default=False, description="Generate AI summaries via Groq.")
    groq_api_key: str = Field(default="", description="Groq API key (if include_ai_summary=True).")
    ai_categorize: bool = Field(
        default=False,
        description=(
            "Use Claude AI to infer the actual sport(s) at each venue from its name and address. "
            "Adds ai_sport_type and ai_confidence to each result. "
            "Especially useful for sport.pitch venues which mix cricket, football, hockey etc."
        ),
    )
    groq_categorize_api_key: str = Field(
        default="",
        description="Groq API key for AI categorization (required if ai_categorize=True). Falls back to GROQ_API_KEY env var.",
    )
    ai_filter_sports: list[str] = Field(
        default_factory=list,
        description=(
            "After AI categorization, keep only venues matching these sport names. "
            "e.g. ['Cricket', 'Football']. Only applies when ai_categorize=true. "
            "Leave empty to return all venues."
        ),
    )
    ai_min_confidence: str = Field(
        default="low",
        description=(
            "Minimum AI confidence level to include in filtered results. "
            "One of: 'high', 'medium', 'low'. Only applies when ai_filter_sports is set."
        ),
    )
    ai_include_unknown: bool = Field(
        default=False,
        description="If true, include venues where AI could not determine the sport type.",
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_api_key() -> str:
    key = os.environ.get("GEOAPIFY_API_KEY", "")
    if not key:
        raise HTTPException(
            status_code=503,
            detail=(
                "GEOAPIFY_API_KEY environment variable is not set. "
                "Get a free key at https://myprojects.geoapify.com/"
            ),
        )
    return key


def validate_sports(sports: list[str]) -> list[str]:
    invalid = [s for s in sports if s not in SPORT_CATEGORY_MAP]
    if invalid:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Unknown sport(s): {invalid}. "
                f"Valid options: {list(SPORT_CATEGORY_MAP.keys())}"
            ),
        )
    return sports


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", tags=["Meta"])
def health():
    """Liveness check."""
    return {"status": "ok"}


@app.get("/sports", response_model=SportListResponse, tags=["Meta"])
def list_sports():
    """Return all available sport categories and their Geoapify category strings."""
    sports = [
        SportCategory(name=name, geoapify_category=cat)
        for name, cat in SPORT_CATEGORY_MAP.items()
    ]
    return SportListResponse(total=len(sports), sports=sports)


@app.get("/geocode", response_model=CityGeocodeOut, tags=["Geocoding"])
def geocode(
    city: Annotated[str, Query(description="City name to geocode, e.g. 'Mumbai'.")],
):
    """
    Resolve a city name to its Geoapify place_id and coordinates.
    Useful for debugging why a city search returns 0 results.
    """
    api_key = get_api_key()
    result: CityGeocode = geocode_city(city, api_key)
    return CityGeocodeOut(
        city=city,
        place_id=result.place_id,
        latitude=result.lat,
        longitude=result.lon,
    )


@app.get("/venues", response_model=VenueListResponse, tags=["Venues"])
def search_venues_get(
    city: Annotated[str, Query(description="City to search in, e.g. 'Delhi'.")],
    sports: Annotated[
        list[str],
        Query(description="Sport names to filter by. Repeat param for multiple."),
    ] = DEFAULT_SPORTS,
    max_results: Annotated[int, Query(ge=1, le=500)] = 50,
):
    """
    Search for sports venues in a city (GET version, good for quick queries).

    Pass `sports` multiple times to filter:
        /venues?city=Delhi&sports=Cricket&sports=Football
    """
    api_key = get_api_key()
    validate_sports(sports)

    venues = scrape_geoapify_venues(
        city=city,
        api_key=api_key,
        sports=sports,
        max_results=max_results,
        page_size=20,
        timeout=20,
        delay=0.3,
    )

    return VenueListResponse(
        city=city,
        total=len(venues),
        venues=[VenueOut.from_venue(v) for v in venues],
    )


@app.post("/venues/search", response_model=VenueListResponse, tags=["Venues"])
def search_venues_post(body: VenueSearchRequest):
    """
    Search for sports venues (POST version with full options).

    When `ai_categorize=true`, AI sport filtering runs inside the scraper —
    only venues where the AI confirms the requested sport are returned.
    No separate filtering step needed on the client side.

    Example body:
    ```json
    {
        "city": "Kolkata",
        "sports": ["Cricket", "Football"],
        "max_results": 100,
        "ai_categorize": true,
        "ai_min_confidence": "medium"
    }
    ```
    """
    api_key = get_api_key()
    validate_sports(body.sports)

    akey = os.environ.get("GROQ_API_KEY", "")
    if body.ai_categorize and not akey:
        raise HTTPException(
            status_code=503,
            detail="GROQ_API_KEY environment variable is not set. Required for ai_categorize=true.",
        )

    # AI filtering is now handled inside scrape_geoapify_venues.
    # When ai_categorize=true, only venues confirmed by the AI are returned.
    # ai_filter_sports defaults to the same sports list the user searched for,
    # so "search for Cricket" → "only return confirmed Cricket venues".
    venues = scrape_geoapify_venues(
        city=body.city,
        api_key=api_key,
        sports=body.sports,
        max_results=body.max_results,
        page_size=body.page_size,
        timeout=20,
        delay=0.3,
        groq_api_key=akey if body.ai_categorize else "",
        ai_filter_sports=(
            body.ai_filter_sports if body.ai_filter_sports
            else body.sports          # default: filter to same sports searched
        ) if body.ai_categorize else None,
        ai_min_confidence=body.ai_min_confidence,
    )

    if body.include_ai_summary:
        groq_key = os.environ.get("GROQ_API_KEY", "")
        if not groq_key:
            raise HTTPException(
                status_code=503,
                detail="GROQ_API_KEY environment variable is not set. Required for include_ai_summary=true.",
            )
        add_ai_summaries(venues, groq_api_key=groq_key, model_id="llama-3.3-70b-versatile", delay=0.3)

    return VenueListResponse(
        city=body.city,
        total=len(venues),
        venues=[VenueOut.from_venue(v) for v in venues],
    )


@app.get("/venues/by-sport", response_model=VenueListResponse, tags=["Venues"])
def venues_by_sport(
    city: Annotated[str, Query(description="City to search in.")],
    ai_sports: Annotated[
        list[str],
        Query(description="AI sport type(s) to filter by. Repeat for multiple. e.g. Cricket, Football."),
    ],
    geoapify_sports: Annotated[
        list[str],
        Query(description="Geoapify sport categories to search. Defaults to all."),
    ] = DEFAULT_SPORTS,
    min_confidence: Annotated[
        str,
        Query(description="Minimum AI confidence: 'high', 'medium', or 'low'."),
    ] = "low",
    include_unknown: Annotated[
        bool,
        Query(description="Include venues where AI could not determine sport type."),
    ] = False,
    max_results: Annotated[int, Query(ge=1, le=500)] = 50,
):
    """
    Search venues and filter results by AI-inferred sport type in one call.

    This endpoint:
      1. Fetches venues from Geoapify for the given city
      2. Runs AI categorization on all results
      3. Returns only venues matching the requested `ai_sports`

    Example — find only cricket venues in Kolkata:
        /venues/by-sport?city=Kolkata&ai_sports=Cricket&min_confidence=medium

    Example — cricket or football, high confidence only:
        /venues/by-sport?city=Kolkata&ai_sports=Cricket&ai_sports=Football&min_confidence=high
    """
    from ai_categorizer import categorize_venues, filter_by_ai_sport

    api_key = get_api_key()
    akey = os.environ.get("GROQ_API_KEY", "")
    if not akey:
        raise HTTPException(
            status_code=503,
            detail="GROQ_API_KEY environment variable is not set. Required for AI sport filtering.",
        )

    validate_sports(geoapify_sports)

    # 1. Fetch from Geoapify
    venues = scrape_geoapify_venues(
        city=city,
        api_key=api_key,
        sports=geoapify_sports,
        max_results=max_results,
        page_size=20,
        timeout=20,
        delay=0.3,
    )

    # 2. AI categorize
    venues = categorize_venues(venues, groq_api_key=akey)

    # 3. Filter by requested sport types
    venues = filter_by_ai_sport(
        venues,
        sports=ai_sports,
        include_unknown=include_unknown,
        min_confidence=min_confidence,
    )

    return VenueListResponse(
        city=city,
        total=len(venues),
        venues=[VenueOut.from_venue(v) for v in venues],
    )


@app.get("/venues/{place_id}", response_model=VenueOut, tags=["Venues"])
def get_venue_by_id(
    place_id: str,
    city: Annotated[str, Query(description="City the venue belongs to (used to scope the search).")],
    sport: Annotated[str | None, Query(description="Sport type to narrow the search.")] = None,
):
    """
    Look up a single venue by its Geoapify place_id.

    Since Geoapify's Places API doesn't support direct lookup by place_id,
    this searches the city (and optional sport) and finds the matching venue.
    """
    api_key = get_api_key()
    sports = [sport] if sport and sport in SPORT_CATEGORY_MAP else DEFAULT_SPORTS

    venues = scrape_geoapify_venues(
        city=city,
        api_key=api_key,
        sports=sports,
        max_results=500,
        page_size=20,
        timeout=20,
        delay=0.3,
    )

    match = next((v for v in venues if v.place_id == place_id), None)
    if not match:
        raise HTTPException(
            status_code=404,
            detail=f"Venue with place_id '{place_id}' not found in {city}.",
        )
    return VenueOut.from_venue(match)
