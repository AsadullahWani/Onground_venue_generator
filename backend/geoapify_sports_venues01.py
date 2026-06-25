"""
Geoapify Places API sports venue scraper with optional Agno AI summaries.

Uses the Geoapify Places API (free tier: 3000 requests/day, no credit card needed)
to collect sports venue data for a city. Sport categories are queried using
Geoapify's native `sport.*` category hierarchy — sport_type is set directly
from the category used, no AI guessing required.

The workflow is:
  1. Geocode the city name → get Geoapify place_id for the city boundary
  2. For each sport category, call Places API with filter=place:<city_place_id>
     and paginate through all results (up to max_results)
  3. Deduplicate by place_id, write CSV + JSON output
  4. (Optional) generate AI summaries via Agno + Groq

Get a free API key at: https://myprojects.geoapify.com/

Usage:
    python geoapify_sports_venues.py "Kolkata" --api-key YOUR_KEY
    python geoapify_sports_venues.py "New Delhi" --api-key YOUR_KEY --output-dir src
    python geoapify_sports_venues.py "Mumbai" --api-key YOUR_KEY --sports sport.pitch sport.stadium --max-results 200
    python geoapify_sports_venues.py "Bangalore" --api-key YOUR_KEY --include-ai-summary --groq-api-key GROQ_KEY
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import requests

# ---------------------------------------------------------------------------
# Geoapify endpoints
# ---------------------------------------------------------------------------

GEOCODE_URL   = "https://api.geoapify.com/v1/geocode/search"
PLACES_URL    = "https://api.geoapify.com/v2/places"

# ---------------------------------------------------------------------------
# Geoapify sport category map
#
# Keys are human-readable sport names (used as sport_type in the output).
# Values are the Geoapify Places API category strings to query.
# Using the parent `sport` category captures everything, or you can target
# individual sub-categories for more precise results.
#
# Full sport sub-categories from Geoapify docs:
#   sport.pitch, sport.stadium, sport.sports_centre, sport.sports_hall,
#   sport.fitness, sport.fitness.gym, sport.fitness.fitness_centre,
#   sport.fitness.fitness_station, sport.golf_course, sport.swimming_pool,
#   sport.ice_rink, sport.track, sport.horse_riding, sport.shooting,
#   sport.skateboard, sport.dojo, sport.dive_centre, sport.fishing
# ---------------------------------------------------------------------------

SPORT_CATEGORY_MAP: dict[str, str] = {
    "Cricket":      "sport.pitch",        # outdoor pitches — cricket dominant in India
    "Football":     "sport.pitch",        # same category, deduplicated by place_id
    "Stadium":      "sport.stadium",
    "Sports Centre":"sport.sports_centre",
    "Sports Hall":  "sport.sports_hall",
    "Swimming":     "sport.swimming_pool",
    "Golf":         "sport.golf_course",
    "Athletics":    "sport.track",
    "Gym":          "sport.fitness.gym",
    "Fitness":      "sport.fitness",
    "Ice Rink":     "sport.ice_rink",
    "Horse Riding": "sport.horse_riding",
    "Shooting":     "sport.shooting",
    "Skateboard":   "sport.skateboard",
    "Martial Arts": "sport.dojo",
    "Diving":       "sport.dive_centre",
    "Fishing":      "sport.fishing",
}

# Default: search all unique categories (deduplicating categories that appear
# under multiple sport names like sport.pitch)
DEFAULT_SPORTS = list(SPORT_CATEGORY_MAP.keys())

# Max results Geoapify returns per page
GEOAPIFY_PAGE_SIZE = 20   # API default; max is 500 but free tier works well at 20


# ---------------------------------------------------------------------------
# Dataclass — mirrors your original Venue structure, adapted for Geoapify
# ---------------------------------------------------------------------------

@dataclass
class Venue:
    place_id: str
    name: str
    city: str
    formatted_address: str = ""
    short_address: str = ""
    latitude: float | None = None
    longitude: float | None = None
    # Geoapify returns a list of categories; we store all of them
    categories: str = ""
    # We set sport_type directly from the search category — no AI needed
    sport_type: str = ""
    search_category: str = ""      # the Geoapify category we queried
    # Contact / meta
    website: str = ""
    phone: str = ""
    opening_hours: str = ""
    # Extra OSM-sourced fields Geoapify exposes
    country: str = ""
    state: str = ""
    postcode: str = ""
    street: str = ""
    distance: float | None = None  # metres from bias point if set
    # Optional AI enrichment
    ai_summary: str = ""
    # AI sport categorization (set by ai_categorizer.categorize_venues)
    ai_sport_type: str = ""    # e.g. "Cricket" or "Football, Cricket" (all detected sports)
    ai_confidence: str = ""    # "high" | "medium" | "low" | "skipped"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def clean(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def slugify(value: str) -> str:
    return "".join(c.lower() if c.isalnum() else "_" for c in value).strip("_")


def sport_name_from_category(category: str, reverse_map: dict[str, str]) -> str:
    """Return the human sport name for a Geoapify category string."""
    for name, cat in reverse_map.items():
        if cat == category:
            return name
    return category.replace("sport.", "").replace("_", " ").title()


# ---------------------------------------------------------------------------
# Step 1 — Geocode city → get Geoapify place_id for city boundary filter
# ---------------------------------------------------------------------------

@dataclass
class CityGeocode:
    place_id: str | None
    lat: float | None
    lon: float | None


def geocode_city(city: str, api_key: str) -> CityGeocode:
    """
    Geocodes a city name and returns its Geoapify place_id plus lat/lon.

    - place_id is used as filter=place:<place_id> to constrain Places API
      results to the city's administrative boundary (most accurate).
    - lat/lon is used as a proximity bias fallback if place_id is unavailable.

    Note: 'type' parameter is NOT sent — Geoapify's geocoding API does not
    support type=city as a filter; omitting it returns the best match.
    """
    params = {
        "text": city,
        "limit": 1,
        "format": "json",
        "apiKey": api_key,
    }
    try:
        resp = requests.get(GEOCODE_URL, params=params, timeout=15)
        resp.raise_for_status()
        results = resp.json().get("results", [])
        if results:
            r = results[0]
            place_id = r.get("place_id")
            lat = r.get("lat")
            lon = r.get("lon")
            print(f"Geocoded '{city}' → place_id: {place_id}  ({lat}, {lon})")
            return CityGeocode(place_id=place_id, lat=lat, lon=lon)
        print(f"Warning: Could not geocode city '{city}'. No results returned.")
        return CityGeocode(place_id=None, lat=None, lon=None)
    except requests.RequestException as exc:
        print(f"Geocoding error for '{city}': {exc}")
        return CityGeocode(place_id=None, lat=None, lon=None)


# ---------------------------------------------------------------------------
# Step 2 — Query Geoapify Places API for a single sport category
# ---------------------------------------------------------------------------

def search_geoapify_for_category(
    category: str,
    city: str,
    geocode: "CityGeocode",
    api_key: str,
    max_results: int,
    page_size: int,
    timeout: int,
    delay: float,
    retries: int = 2,
) -> list[Venue]:
    """
    Fetches all venues of a given Geoapify sport category within the city.
    Paginates using the `offset` parameter until max_results is reached
    or no more results are returned.

    Spatial strategy (in priority order):
      1. filter=place:<place_id>  — constrains to city administrative boundary (best)
      2. bias=proximity:lon,lat   — biases toward city centre, no hard boundary (fallback)
      Neither conditions=named nor type=city are valid Geoapify parameters and are omitted.
    """
    sport_name = sport_name_from_category(category, SPORT_CATEGORY_MAP)
    venues: list[Venue] = []
    offset = 0

    # Build the spatial filter: prefer city boundary, fall back to proximity bias
    if geocode.place_id:
        spatial_filter = f"place:{geocode.place_id}"
        bias = None
    elif geocode.lat is not None and geocode.lon is not None:
        # proximity bias: lon,lat order (Geoapify convention)
        spatial_filter = None
        bias = f"proximity:{geocode.lon},{geocode.lat}"
        print(f"  Info: using proximity bias ({geocode.lat}, {geocode.lon}) — results near {city} centre.")
    else:
        print(f"  Warning: no geocode data for '{city}'. Cannot spatially constrain query.")
        return []

    while len(venues) < max_results:
        fetch_size = min(page_size, max_results - len(venues))

        params: dict[str, Any] = {
            "categories": category,
            "limit": fetch_size,
            "offset": offset,
            "lang": "en",
            "apiKey": api_key,
        }
        if spatial_filter:
            params["filter"] = spatial_filter
        if bias:
            params["bias"] = bias

        success = False
        for attempt in range(1, retries + 2):
            try:
                resp = requests.get(PLACES_URL, params=params, timeout=timeout)
                if resp.status_code == 429:
                    wait = 30 * attempt
                    print(f"  Rate limited. Waiting {wait}s (attempt {attempt})...")
                    time.sleep(wait)
                    continue
                resp.raise_for_status()
                success = True
                break
            except requests.RequestException as exc:
                if attempt <= retries:
                    print(f"  Request failed ({exc}). Retrying in 10s...")
                    time.sleep(10)
                else:
                    print(f"  Failed to fetch {category} in {city}: {exc}")
                    return venues

        if not success:
            break

        data = resp.json()
        features = data.get("features", [])
        if not features:
            break   # no more results

        for feature in features:
            props = feature.get("properties", {})
            name = clean(props.get("name", ""))
            if not name:
                continue

            geo = feature.get("geometry", {})
            coords = geo.get("coordinates", [None, None]) if geo.get("type") == "Point" else [None, None]
            lon, lat = coords[0], coords[1]

            # categories is a list like ["sport.pitch", "sport"]
            raw_cats = props.get("categories") or []
            categories_str = ", ".join(raw_cats)

            venue = Venue(
                place_id=clean(props.get("place_id", "")),
                name=name,
                city=city,
                formatted_address=clean(props.get("formatted", "")),
                short_address=clean(props.get("address_line1", "")),
                latitude=lat,
                longitude=lon,
                categories=categories_str,
                sport_type=sport_name,        # set directly from search — no AI needed
                search_category=category,
                website=clean(props.get("website", "") or props.get("contact:website", "")),
                phone=clean(props.get("phone", "") or props.get("contact:phone", "")),
                opening_hours=clean(props.get("opening_hours", "")),
                country=clean(props.get("country", "")),
                state=clean(props.get("state", "")),
                postcode=clean(props.get("postcode", "")),
                street=clean(props.get("street", "")),
                distance=props.get("distance"),
            )
            venues.append(venue)

        offset += len(features)

        # If fewer results than requested, we've hit the end
        if len(features) < fetch_size:
            break

        if delay:
            time.sleep(delay)

    return venues


# ---------------------------------------------------------------------------
# Step 3 — Scrape all sport categories, deduplicate
# ---------------------------------------------------------------------------

def scrape_geoapify_venues(
    city: str,
    api_key: str,
    sports: list[str],
    max_results: int,
    page_size: int,
    timeout: int,
    delay: float,
    # --- AI strict filtering (optional) ---
    # When groq_api_key + ai_filter_sports are both provided, AI categorization
    # runs inside this function and non-matching venues are dropped before returning.
    # This means the returned list only contains venues where the AI confirmed
    # the requested sport — not just anything Geoapify returned for that category.
    groq_api_key: str = "",
    ai_filter_sports: list[str] | None = None,
    ai_min_confidence: str = "low",
    ai_batch_size: int = 20,
) -> list[Venue]:
    """
    Resolves the city boundary, fetches venues from Geoapify, deduplicates,
    then optionally runs AI sport categorization and filters the results so
    only venues matching the requested sports are returned.

    AI filtering behaviour:
      - Only runs when groq_api_key is provided AND ai_filter_sports is non-empty.
      - Uses the same sports list you searched for as the filter target by default
        (pass ai_filter_sports explicitly to override).
      - Venues where the AI is not confident are kept or dropped based on
        ai_min_confidence ("high" | "medium" | "low").
      - Venues where categorization is unambiguous from the Geoapify category
        alone (e.g. swimming pools, golf courses) skip the AI call entirely.

    Without groq_api_key the function behaves exactly as before — Geoapify
    results are returned as-is with no AI filtering applied.
    """
    # Resolve city → place_id + lat/lon once
    geocode = geocode_city(city, api_key)

    # Build set of unique Geoapify categories to actually query.
    # Multiple sport names may map to the same category (e.g. Cricket + Football
    # both map to sport.pitch) — we only query each category once.
    queried_categories: dict[str, str] = {}   # category → first sport_name seen
    for sport_name in sports:
        category = SPORT_CATEGORY_MAP.get(sport_name)
        if category and category not in queried_categories:
            queried_categories[category] = sport_name

    by_place_id: dict[str, Venue] = {}

    for category, sport_name in queried_categories.items():
        print(f"Searching Geoapify: [{category}] '{sport_name}' venues in {city}...")
        batch = search_geoapify_for_category(
            category=category,
            city=city,
            geocode=geocode,
            api_key=api_key,
            max_results=max_results,
            page_size=page_size,
            timeout=timeout,
            delay=delay,
        )
        print(f"  → {len(batch)} raw results from Geoapify")

        for venue in batch:
            key = venue.place_id or f"{venue.name.lower()}|{venue.latitude}|{venue.longitude}"
            if key not in by_place_id:
                by_place_id[key] = venue
            else:
                # Same venue found under another category — merge sport types
                existing = by_place_id[key]
                existing_sports = set(existing.sport_type.split(", "))
                existing_sports.add(sport_name)
                existing.sport_type = ", ".join(sorted(existing_sports))

        if len(by_place_id) >= max_results:
            print(f"Reached max_results ({max_results}). Stopping Geoapify search early.")
            break

    all_venues: list[Venue] = sorted(by_place_id.values(), key=lambda v: v.name.lower())

    # ------------------------------------------------------------------
    # AI strict filtering — runs inside the scraper so callers always
    # receive a pre-filtered list that matches the requested sports.
    # ------------------------------------------------------------------
    should_ai_filter = groq_api_key and ai_filter_sports

    if should_ai_filter:
        from ai_categorizer import categorize_venues, filter_by_ai_sport

        # Default: use the same sport names the user searched for
        target_sports = ai_filter_sports or sports

        print(f"\nRunning AI strict filtering: {len(all_venues)} venues → keeping {target_sports}...")
        all_venues = categorize_venues(
            all_venues,
            groq_api_key=groq_api_key,
            batch_size=ai_batch_size,
        )

        before = len(all_venues)
        all_venues = filter_by_ai_sport(
            all_venues,
            sports=target_sports,
            min_confidence=ai_min_confidence,
            include_unknown=False,   # strict — drop anything AI couldn't classify
        )
        print(f"AI strict filter: {before} → {len(all_venues)} venues matched {target_sports} "
              f"(min_confidence={ai_min_confidence})")
    elif groq_api_key and not ai_filter_sports:
        # groq key provided but no filter requested — still run categorization
        # so ai_sport_type is populated, but don't drop any venues
        from ai_categorizer import categorize_venues
        print(f"\nRunning AI categorization (no filter)...")
        all_venues = categorize_venues(all_venues, groq_api_key=groq_api_key, batch_size=ai_batch_size)

    return all_venues[:max_results]


# ---------------------------------------------------------------------------
# Optional AI summary (Agno + Groq) — same pattern as original script
# ---------------------------------------------------------------------------

def venue_summary_prompt(venue: Venue) -> str:
    payload = {
        "name": venue.name,
        "address": venue.formatted_address,
        "sport_type": venue.sport_type,
        "categories": venue.categories,
        "opening_hours": venue.opening_hours,
        "country": venue.country,
        "state": venue.state,
    }
    return f"""
Analyze this sports venue and write a short summary.

Return JSON only. No markdown. No preamble.

Format:
{{
    "summary": "35–50 word summary covering venue type, sport, and notable attributes"
}}

Venue data:
{json.dumps(payload)}
"""


def _agent_text(response: Any) -> str:
    if hasattr(response, "get_content_as_string"):
        return clean(response.get_content_as_string())
    if hasattr(response, "content"):
        return clean(response.content)
    if hasattr(response, "output"):
        return clean(response.output)
    return clean(str(response))


def add_ai_summaries(
    venues: list[Venue],
    groq_api_key: str,
    model_id: str,
    delay: float,
) -> None:
    try:
        from agno.agent import Agent
        from agno.models.groq import Groq
    except ImportError:
        print("agno not installed. Skipping AI summaries. Run: pip install agno")
        return

    agent = Agent(
        model=Groq(id=model_id, api_key=groq_api_key),
        instructions=[
            "You summarize sports venues for a dataset.",
            "Use only the venue data provided. Do not invent attributes.",
            "Write one concise paragraph, 35 to 50 words.",
            "Return JSON only. No markdown.",
            '{"summary": "Short venue summary"}',
        ],
        markdown=False,
    )

    for index, venue in enumerate(venues, start=1):
        print(f"Generating AI summary {index}/{len(venues)}: {venue.name}")
        try:
            response = agent.run(venue_summary_prompt(venue))
            text = _agent_text(response)
            clean_text = text.replace("```json", "").replace("```", "").strip()
            data = json.loads(clean_text)
            venue.ai_summary = data.get("summary", clean_text)
        except Exception as exc:
            venue.ai_summary = f"AI summary failed: {exc}"

        if delay:
            time.sleep(delay)


# ---------------------------------------------------------------------------
# Output — CSV + JSON
# ---------------------------------------------------------------------------

def save_outputs(venues: list[Venue], city: str, output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    base_name = f"geoapify_sports_venues_{slugify(city)}"
    csv_path  = output_dir / f"{base_name}.csv"
    json_path = output_dir / f"{base_name}.json"
    rows = [asdict(venue) for venue in venues]

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(Venue.__dataclass_fields__))
        writer.writeheader()
        writer.writerows(rows)

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)

    return csv_path, json_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Collect sports venue data from Geoapify Places API. "
            "Free tier: 3000 requests/day. Get a key at https://myprojects.geoapify.com/"
        )
    )
    parser.add_argument("city", help="City name, e.g. 'Kolkata' or 'New Delhi'.")
    parser.add_argument(
        "--api-key",
        default=os.environ.get("GEOAPIFY_API_KEY", ""),
        help="Geoapify API key. Can also be set via GEOAPIFY_API_KEY env var.",
    )
    parser.add_argument(
        "--sports",
        nargs="+",
        default=DEFAULT_SPORTS,
        choices=list(SPORT_CATEGORY_MAP.keys()),
        metavar="SPORT",
        help=(
            f"Sport names to search. Choices: {', '.join(SPORT_CATEGORY_MAP.keys())}. "
            "Default: all sports."
        ),
    )
    parser.add_argument(
        "--max-results",
        type=int,
        default=200,
        help="Maximum unique venues to return across all sport categories. Default: 200.",
    )
    parser.add_argument(
        "--page-size",
        type=int,
        default=20,
        help="Results per Geoapify API request (1–500). Default: 20 (free-tier friendly).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("."),
        help="Directory for CSV/JSON output. Default: current directory.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=20,
        help="HTTP request timeout in seconds. Default: 20.",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.5,
        help="Delay in seconds between API requests (rate-limit courtesy). Default: 0.5.",
    )
    parser.add_argument(
        "--ai-categorize",
        action="store_true",
        help=(
            "Use Claude AI to infer the actual sport(s) at each venue from its name "
            "and address. Adds ai_sport_type and ai_confidence columns. "
            "Requires --anthropic-api-key."
        ),
    )
    parser.add_argument(
        "--anthropic-api-key",
        default=os.environ.get("GROQ_API_KEY", ""),
        help="Groq API key for AI sport categorization. Can also be set via GROQ_API_KEY env var.",
    )
    parser.add_argument(
        "--ai-filter-sports",
        nargs="+",
        default=[],
        metavar="SPORT",
        help=(
            "After AI categorization, keep only venues matching these sport names. "
            "e.g. --ai-filter-sports Cricket Football. Requires --ai-categorize."
        ),
    )
    parser.add_argument(
        "--ai-min-confidence",
        default="low",
        choices=["high", "medium", "low"],
        help="Minimum AI confidence level when filtering. Default: low (keep all).",
    )
    parser.add_argument(
        "--ai-include-unknown",
        action="store_true",
        help="When filtering by sport, also include venues where AI could not determine the sport.",
    )
    parser.add_argument(
        "--include-ai-summary",
        action="store_true",
        help="Generate an ai_summary column using Agno + Groq.",
    )
    parser.add_argument(
        "--groq-api-key",
        default=os.environ.get("GROQ_API_KEY", ""),
        help="Groq API key for AI summaries. Can also be set via GROQ_API_KEY env var.",
    )
    parser.add_argument(
        "--ai-model",
        default="llama-3.3-70b-versatile",
        help="Groq model ID for AI summaries. Default: llama-3.3-70b-versatile.",
    )
    parser.add_argument(
        "--ai-delay",
        type=float,
        default=0.5,
        help="Delay between AI summary calls in seconds. Default: 0.5.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    if not args.api_key:
        print(
            "Error: Geoapify API key is required.\n"
            "Pass --api-key YOUR_KEY or set the GEOAPIFY_API_KEY environment variable.\n"
            "Get a free key (no credit card) at: https://myprojects.geoapify.com/"
        )
        return

    venues = scrape_geoapify_venues(
        city=args.city,
        api_key=args.api_key,
        sports=args.sports,
        max_results=max(1, args.max_results),
        page_size=max(1, min(500, args.page_size)),
        timeout=max(5, args.timeout),
        delay=max(0.0, args.delay),
        # AI strict filtering — applied inside the scraper before returning
        groq_api_key=args.anthropic_api_key if args.ai_categorize else "",
        ai_filter_sports=args.ai_filter_sports or args.sports if args.ai_categorize else None,
        ai_min_confidence=args.ai_min_confidence,
    )

    if not venues:
        print("No venues found. Try a different city name or sport list.")
        return

    if args.include_ai_summary:
        if not args.groq_api_key:
            print("Warning: --groq-api-key not provided. Skipping AI summaries.")
        else:
            print(f"\nGenerating AI summaries for {len(venues)} venues...")
            add_ai_summaries(
                venues,
                groq_api_key=args.groq_api_key,
                model_id=args.ai_model,
                delay=max(0.0, args.ai_delay),
            )

    csv_path, json_path = save_outputs(venues, args.city, args.output_dir)

    print(f"\nSaved {len(venues)} venues.")
    print(f"CSV  : {csv_path}")
    print(f"JSON : {json_path}")

    print("\nSample (first 10):")
    for v in venues[:10]:
        print(f"  - {v.name} [{v.sport_type}] — {v.short_address or v.formatted_address or 'no address'}")


# ---------------------------------------------------------------------------
# Importable entry point — drop-in replacement for the original fetch_venues()
# ---------------------------------------------------------------------------

def fetch_venues(
    city: str,
    api_key: str = "",
    sports: list[str] | None = None,
    include_ai_summary: bool = False,
    groq_api_key: str = "",
    ai_categorize: bool = False,
    groq_categorize_api_key: str = "",
    ai_filter_sports: list[str] | None = None,
    ai_min_confidence: str = "low",
    max_results: int = 100,
) -> list[Venue]:
    """
    Importable entry point.

    When groq_categorize_api_key is provided, AI categorization runs inside
    scrape_geoapify_venues and only venues matching `ai_filter_sports` (or
    `sports` if not specified) are returned. No separate filtering step needed.

    Args:
        city:                    City name to search in.
        api_key:                 Geoapify API key. Falls back to GEOAPIFY_API_KEY env var.
        sports:                  Sport names from SPORT_CATEGORY_MAP. Defaults to all.
        include_ai_summary:      Generate AI summaries via Agno + Groq.
        groq_api_key:            Groq API key for AI summaries.
        ai_categorize:           Run AI sport categorization.
        groq_categorize_api_key: Groq API key for AI categorization.
        ai_filter_sports:        Sports to keep after AI categorization.
                                 Defaults to the same list as `sports`.
        ai_min_confidence:       Minimum AI confidence: "high" | "medium" | "low".
        max_results:             Max venues to return.
    """
    key = api_key or os.environ.get("GEOAPIFY_API_KEY", "")
    if not key:
        raise ValueError("Geoapify API key required. Pass api_key= or set GEOAPIFY_API_KEY.")

    search_sports = sports or DEFAULT_SPORTS
    akey = groq_categorize_api_key or os.environ.get("GROQ_API_KEY", "")

    venues = scrape_geoapify_venues(
        city=city,
        api_key=key,
        sports=search_sports,
        max_results=max_results,
        page_size=20,
        timeout=20,
        delay=0.3,
        # Pass AI params directly — filtering happens inside the scraper
        groq_api_key=akey if ai_categorize else "",
        ai_filter_sports=ai_filter_sports or (search_sports if ai_categorize else None),
        ai_min_confidence=ai_min_confidence,
    )

    if include_ai_summary and groq_api_key:
        add_ai_summaries(
            venues,
            groq_api_key=groq_api_key,
            model_id="llama-3.3-70b-versatile",
            delay=0.5,
        )

    return venues


if __name__ == "__main__":
    main()
