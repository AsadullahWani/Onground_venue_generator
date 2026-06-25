"""
ai_categorizer.py — AI sport categorization for Geoapify venue data.

Uses Agno + Groq to infer the actual sport(s) played at a venue from its
name, address, and Geoapify categories.

This is useful because Geoapify's sport.pitch category returns cricket grounds,
football fields, hockey pitches, and kabaddi courts all mixed together —
the search category alone cannot distinguish them.

What this module does:
  - Creates one Agno Agent (reused across all batches — no repeated init cost)
  - Uses Agno's response_model (Pydantic) for structured output — no JSON parsing
  - Batch-processes venues in groups to minimise API calls
  - Returns structured output: primary_sport, all_sports[], confidence, reasoning
  - Adds two fields to each Venue: ai_sport_type, ai_confidence
  - Skips venues where Geoapify's category is already unambiguous
    (e.g. sport.swimming_pool, sport.golf_course, sport.ice_rink)

Requirements:
    pip install agno groq

Usage:
    from ai_categorizer import categorize_venues
    venues = categorize_venues(venues, groq_api_key="your-groq-key")
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from typing import Any, List

# ---------------------------------------------------------------------------
# Agno imports
# ---------------------------------------------------------------------------
from agno.agent import Agent
from agno.models.groq import Groq
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Geoapify categories that are already unambiguous — skip AI for these
# ---------------------------------------------------------------------------

UNAMBIGUOUS_CATEGORIES: dict[str, str] = {
    "sport.swimming_pool": "Swimming",
    "sport.golf_course":   "Golf",
    "sport.ice_rink":      "Ice Rink",
    "sport.horse_riding":  "Horse Riding",
    "sport.shooting":      "Shooting",
    "sport.skateboard":    "Skateboard",
    "sport.dojo":          "Martial Arts",
    "sport.dive_centre":   "Diving",
    "sport.fishing":       "Fishing",
    "sport.fitness.gym":   "Gym",
}

# Categories where AI adds real value
AMBIGUOUS_CATEGORIES: set[str] = {
    "sport.pitch",          # cricket / football / hockey / kabaddi etc.
    "sport.stadium",        # could host multiple sports
    "sport.sports_centre",  # definitely multi-sport
    "sport.sports_hall",    # indoor multi-sport
    "sport.track",          # athletics / cycling / motor racing
    "sport.fitness",        # gym / yoga / crossfit etc.
}

# Default Groq model — fast and free-tier friendly
DEFAULT_MODEL = "llama-3.3-70b-versatile"

# ---------------------------------------------------------------------------
# Pydantic models for Agno structured output
# ---------------------------------------------------------------------------

class VenueCategorization(BaseModel):
    """Categorization result for a single venue."""
    place_id: str = Field(description="The place_id from the input record.")
    primary_sport: str = Field(description="Single most likely sport, title-cased.")
    all_sports: List[str] = Field(description="All sports detected, ordered by likelihood.")
    confidence: str = Field(description="'high', 'medium', or 'low'.")
    reasoning: str = Field(description="One sentence explaining the classification.")


class VenueCategorizationList(BaseModel):
    """Wrapper so Agno can return a list of categorizations in one call."""
    venues: List[VenueCategorization] = Field(
        description="One categorization per input venue, same order as input."
    )


# ---------------------------------------------------------------------------
# Result dataclass (public interface — unchanged from before)
# ---------------------------------------------------------------------------

@dataclass
class SportCategorization:
    place_id: str
    primary_sport: str
    all_sports: list[str]
    confidence: str
    reasoning: str


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a sports venue classifier. Given a list of venue records, determine
what sport(s) are played at each venue.

Rules:
- Use the venue name as the strongest signal.
  Examples: "Eden Gardens" → Cricket. "Salt Lake Stadium" → Football.
- Use address and city as supporting context
  (India → cricket likely, Brazil → football likely).
- A venue may host multiple sports. List all plausible ones, most likely first.
- Use standard sport names: Cricket, Football, Hockey, Kabaddi, Basketball,
  Volleyball, Badminton, Tennis, Athletics, Swimming, Boxing, Wrestling,
  Cycling, Rugby, Baseball, Gym, Yoga, Squash, Table Tennis, Multi-Sport.
- Confidence:
    "high"   — name is unambiguous (e.g. "XYZ Cricket Ground")
    "medium" — name strongly suggests but is not explicit (e.g. "XYZ Stadium" in India)
    "low"    — name is generic (e.g. "Sports Complex", "Playground")
- Return one result per venue, same order as the input list.
"""


# ---------------------------------------------------------------------------
# Agent factory — call once per session, reuse across batches
# ---------------------------------------------------------------------------

def _make_agent(groq_api_key: str, model_id: str) -> Agent:
    """
    Creates and returns a single Agno Agent configured for venue categorization.

    The agent uses response_model=VenueCategorizationList so Agno handles
    all JSON parsing and validation — no manual parsing needed.
    """
    return Agent(
        model=Groq(id=model_id, api_key=groq_api_key),
        description="Sports venue classifier that identifies the sport(s) played at venues.",
        instructions=[SYSTEM_PROMPT],
        output_schema=VenueCategorizationList,
        markdown=False,
    )


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def _build_prompt(venues: list[dict[str, Any]]) -> str:
    """Build the user message: a JSON array of slim venue records."""
    records = [
        {
            "place_id":        v.get("place_id", ""),
            "name":            v.get("name", ""),
            "address":         v.get("formatted_address") or v.get("short_address", ""),
            "city":            v.get("city", ""),
            "categories":      v.get("categories", ""),
            "search_category": v.get("search_category", ""),
        }
        for v in venues
    ]
    return (
        "Classify the sport(s) for each venue below.\n"
        "Return one result per venue, same order as input.\n\n"
        + json.dumps(records, ensure_ascii=False, indent=2)
    )


# ---------------------------------------------------------------------------
# Single batch call via Agno
# ---------------------------------------------------------------------------

def _categorize_batch(
    venues: list[dict[str, Any]],
    agent: Agent,
    retries: int = 2,
) -> list[SportCategorization]:
    """
    Send one batch of venue dicts to Groq via Agno and return categorizations.

    Agno's response_model handles structured output — agent.run() returns a
    RunResponse whose .content is already a VenueCategorizationList instance.
    """
    prompt = _build_prompt(venues)

    for attempt in range(1, retries + 2):
        try:
            response = agent.run(prompt)
            break
        except Exception as exc:
            err = str(exc)
            if "429" in err or "rate" in err.lower():
                wait = 20 * attempt
                print(f"  Rate limited by Groq. Waiting {wait}s (attempt {attempt})...")
                time.sleep(wait)
            elif attempt <= retries:
                print(f"  Agno/Groq error ({exc}). Retrying in 10s...")
                time.sleep(10)
            else:
                print(f"  Agno/Groq failed after {retries + 1} attempts: {exc}")
                return _fallback_categorizations(venues)

    # response.content is a VenueCategorizationList (Pydantic model)
    # because we set response_model on the Agent.
    result: VenueCategorizationList | None = None

    if isinstance(response.content, VenueCategorizationList):
        result = response.content
    elif isinstance(response.content, str):
        # Fallback: model returned plain text despite response_model — parse manually
        try:
            raw = response.content.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
            data = json.loads(raw)
            # Handle both {"venues": [...]} and bare [...]
            if isinstance(data, list):
                data = {"venues": data}
            result = VenueCategorizationList(**data)
        except Exception as parse_exc:
            print(f"  Warning: Could not parse response as JSON: {parse_exc}")
            return _fallback_categorizations(venues)
    else:
        print(f"  Warning: Unexpected response type {type(response.content)}. Using fallbacks.")
        return _fallback_categorizations(venues)

    return [
        SportCategorization(
            place_id=v.place_id,
            primary_sport=v.primary_sport,
            all_sports=v.all_sports or [v.primary_sport],
            confidence=v.confidence,
            reasoning=v.reasoning,
        )
        for v in result.venues
    ]


def _fallback_categorizations(venues: list[dict[str, Any]]) -> list[SportCategorization]:
    """Return low-confidence fallbacks when the API call fails entirely."""
    return [
        SportCategorization(
            place_id=v.get("place_id", ""),
            primary_sport="Unknown",
            all_sports=["Unknown"],
            confidence="low",
            reasoning="AI categorization failed.",
        )
        for v in venues
    ]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def categorize_venues(
    venues: list[Any],
    groq_api_key: str,
    model_id: str = DEFAULT_MODEL,
    batch_size: int = 20,
    delay: float = 0.5,
    skip_unambiguous: bool = True,
) -> list[Any]:
    """
    Run AI sport categorization on a list of Venue objects using Agno + Groq.

    Adds two attributes to each Venue in-place:
        venue.ai_sport_type  — e.g. "Cricket" or "Football, Cricket"
        venue.ai_confidence  — "high" | "medium" | "low" | "skipped"

    Args:
        venues:           List of Venue dataclass instances.
        groq_api_key:     Groq API key (get one free at console.groq.com).
        model_id:         Groq model to use. Default: llama-3.3-70b-versatile.
        batch_size:       Venues per Agno/Groq call. Keep ≤ 20 for best results.
        delay:            Seconds between batches (rate-limit courtesy).
        skip_unambiguous: Skip venues with already-clear Geoapify categories.

    Returns:
        The same list with ai_sport_type and ai_confidence set on each venue.
    """
    _ensure_ai_fields(venues)

    # Split: already known vs needs AI
    to_categorize: list[Any] = []
    for v in venues:
        cat = getattr(v, "search_category", "")
        if skip_unambiguous and cat in UNAMBIGUOUS_CATEGORIES:
            v.ai_sport_type = UNAMBIGUOUS_CATEGORIES[cat]
            v.ai_confidence = "skipped"
        else:
            to_categorize.append(v)

    if not to_categorize:
        print("All venues had unambiguous categories. No AI categorization needed.")
        return venues

    skipped = len(venues) - len(to_categorize)
    print(f"AI categorizing {len(to_categorize)} venues "
          f"({skipped} skipped as unambiguous)...")

    # Create agent once — reused across all batches
    agent = _make_agent(groq_api_key=groq_api_key, model_id=model_id)

    total_batches = math.ceil(len(to_categorize) / batch_size)

    for batch_idx in range(total_batches):
        start = batch_idx * batch_size
        batch = to_categorize[start: start + batch_size]

        print(f"  Batch {batch_idx + 1}/{total_batches} ({len(batch)} venues)...")

        batch_dicts = [_venue_to_dict(v) for v in batch]
        categorizations = _categorize_batch(batch_dicts, agent)

        # Map results back by place_id
        result_map = {c.place_id: c for c in categorizations}

        for venue in batch:
            pid = getattr(venue, "place_id", "")
            cat = result_map.get(pid)
            if cat:
                venue.ai_sport_type = ", ".join(cat.all_sports)
                venue.ai_confidence = cat.confidence
            else:
                venue.ai_sport_type = "Unknown"
                venue.ai_confidence = "low"

        if delay and batch_idx < total_batches - 1:
            time.sleep(delay)

    return venues


def filter_by_ai_sport(
    venues: list[Any],
    sports: list[str],
    include_unknown: bool = False,
    min_confidence: str = "low",
) -> list[Any]:
    """
    Filter a list of already-categorized venues by ai_sport_type.

    Args:
        venues:          Venues already processed by categorize_venues().
        sports:          Sport names to keep. Case-insensitive. Partial match
                         supported — "Cricket" matches "Cricket, Football".
        include_unknown: Keep venues where AI returned "Unknown". Default False.
        min_confidence:  "high" → only high; "medium" → high+medium; "low" → all.
                         "skipped" (unambiguous) is always included.

    Returns:
        Filtered list preserving original order.

    Example:
        venues = categorize_venues(venues, groq_api_key=key)
        cricket = filter_by_ai_sport(venues, ["Cricket"], min_confidence="medium")
    """
    CONFIDENCE_RANK = {"high": 3, "medium": 2, "low": 1, "skipped": 3}
    min_rank = CONFIDENCE_RANK.get(min_confidence, 1)
    requested = {s.strip().lower() for s in sports}

    filtered = []
    for v in venues:
        ai_sport   = getattr(v, "ai_sport_type", "").strip()
        confidence = getattr(v, "ai_confidence", "low").strip()

        if not ai_sport or ai_sport.lower() == "unknown":
            if include_unknown:
                filtered.append(v)
            continue

        if CONFIDENCE_RANK.get(confidence, 1) < min_rank:
            continue

        venue_sports = {s.strip().lower() for s in ai_sport.split(",")}
        if venue_sports & requested:
            filtered.append(v)

    return filtered


def categorize_single(
    venue: Any,
    groq_api_key: str,
    model_id: str = DEFAULT_MODEL,
) -> SportCategorization:
    """
    Categorize a single venue. Returns a SportCategorization result.
    Useful for the FastAPI single-venue endpoint.
    """
    _ensure_ai_fields([venue])
    agent = _make_agent(groq_api_key=groq_api_key, model_id=model_id)
    results = _categorize_batch([_venue_to_dict(venue)], agent)
    return results[0] if results else SportCategorization(
        place_id=getattr(venue, "place_id", ""),
        primary_sport="Unknown",
        all_sports=["Unknown"],
        confidence="low",
        reasoning="Categorization returned no results.",
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _venue_to_dict(venue: Any) -> dict[str, Any]:
    try:
        from dataclasses import asdict
        return asdict(venue)
    except TypeError:
        return venue.__dict__.copy()


def _ensure_ai_fields(venues: list[Any]) -> None:
    for v in venues:
        if not hasattr(v, "ai_sport_type"):
            try:
                object.__setattr__(v, "ai_sport_type", "")
            except AttributeError:
                setattr(v, "ai_sport_type", "")
        if not hasattr(v, "ai_confidence"):
            try:
                object.__setattr__(v, "ai_confidence", "")
            except AttributeError:
                setattr(v, "ai_confidence", "")
