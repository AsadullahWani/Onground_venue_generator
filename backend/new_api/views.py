import os

from django.conf import settings
from rest_framework import status
from rest_framework.response import Response
from django.http import HttpResponse
from rest_framework.views import APIView

from .geoapify_sports_venues import (
    DEFAULT_SPORTS,
    SPORT_CATEGORY_MAP,
    add_ai_summaries,
    geocode_city,
    scrape_geoapify_venues,
)

from .serializers import (
    CityGeocodeOutSerializer,
    SportCategorySerializer,
    SportListResponseSerializer,
    VenueOutSerializer,
    VenueListResponseSerializer,
    VenueSearchRequestSerializer,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_api_key():
    key = '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'
    if not key:
        return None, Response(
            {"detail": "GEOAPIFY_API_KEY is not set."},
            status=status.HTTP_503_SERVICE_UNAVAILABLE,
        )
    return key, None


def get_groq_key():
    key = settings.GROQ_API_KEY
    if not key:
        return None, Response(
            {"detail": "GROQ_API_KEY is not set."},
            status=status.HTTP_503_SERVICE_UNAVAILABLE,
        )
    return key, None


def validate_sports(sports: list[str]):
    invalid = [s for s in sports if s not in SPORT_CATEGORY_MAP]
    if invalid:
        return Response(
            {"detail": f"Unknown sport(s): {invalid}. Valid: {list(SPORT_CATEGORY_MAP.keys())}"},
            status=status.HTTP_422_UNPROCESSABLE_ENTITY,
        )
    return None


def serialize_venues(venues) -> list[dict]:
    return [VenueOutSerializer.from_venue(v) for v in venues]


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------

class HealthView(APIView):
    def get(self, request):
        return Response({"status": "ok"})


# ---------------------------------------------------------------------------
# GET /sports
# ---------------------------------------------------------------------------

class SportListView(APIView):
    def get(self, request):
        sports = [
            {"name": name, "geoapify_category": cat}
            for name, cat in SPORT_CATEGORY_MAP.items()
        ]
        return Response({"total": len(sports), "sports": sports})


# ---------------------------------------------------------------------------
# GET /geocode?city=...
# ---------------------------------------------------------------------------

class GeocodeView(APIView):
    def get(self, request):
        city = request.query_params.get("city")
        if not city:
            return Response(
                {"detail": "Query param 'city' is required."},
                status=status.HTTP_422_UNPROCESSABLE_ENTITY,
            )

        api_key, err = get_api_key()
        if err:
            return err

        result = geocode_city(city, api_key)
        return Response({
            "city": city,
            "place_id": result.place_id,
            "latitude": result.lat,
            "longitude": result.lon,
        })


# ---------------------------------------------------------------------------
# GET /venues?city=...&sports=...
# ---------------------------------------------------------------------------

class VenueListView(APIView):
    def get(self, request):
        city = request.query_params.get("city")
        if not city:
            return Response(
                {"detail": "Query param 'city' is required."},
                status=status.HTTP_422_UNPROCESSABLE_ENTITY,
            )

        sports = request.query_params.getlist("sports") or list(DEFAULT_SPORTS)
        try:
            max_results = int(request.query_params.get("max_results", 50))
        except ValueError:
            return Response(
                {"detail": "max_results must be an integer."},
                status=status.HTTP_422_UNPROCESSABLE_ENTITY,
            )

        api_key, err = get_api_key()
        if err:
            return err

        sport_err = validate_sports(sports)
        if sport_err:
            return sport_err

        venues = scrape_geoapify_venues(
            city=city,
            api_key=api_key,
            sports=sports,
            max_results=max_results,
            page_size=20,
            timeout=20,
            delay=0.3,
        )

        return Response({
            "city": city,
            "total": len(venues),
            "venues": serialize_venues(venues),
        })


# ---------------------------------------------------------------------------
# POST /venues/search
# ---------------------------------------------------------------------------

class VenueSearchView(APIView):
    def post(self, request):
        serializer = VenueSearchRequestSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_422_UNPROCESSABLE_ENTITY)

        data = serializer.validated_data
        api_key, err = get_api_key()
        if err:
            return err

        venues = scrape_geoapify_venues(
            city=data["city"],
            api_key=api_key,
            sports=data["sports"],
            max_results=data["max_results"],
            page_size=data["page_size"],
            timeout=20,
            delay=0.3,
        )

        if data["ai_categorize"]:
            from ai_categorizer import categorize_venues, filter_by_ai_sport
            groq_key, err = get_groq_key()
            if err:
                return err

            venues = categorize_venues(venues, groq_api_key=groq_key)

            if data["ai_filter_sports"]:
                before = len(venues)
                venues = filter_by_ai_sport(
                    venues,
                    sports=data["ai_filter_sports"],
                    include_unknown=data["ai_include_unknown"],
                    min_confidence=data["ai_min_confidence"],
                )
                print(f"AI filter: {before} → {len(venues)} venues")

        if data["include_ai_summary"]:
            groq_key, err = get_groq_key()
            if err:
                return err
            add_ai_summaries(venues, groq_api_key=groq_key, model_id="groq/compound", delay=35.0)

        return Response({
            "city": data["city"],
            "total": len(venues),
            "venues": serialize_venues(venues),
        })


# ---------------------------------------------------------------------------
# GET /venues/by-sport
# ---------------------------------------------------------------------------

class VenueBySportView(APIView):
    def get(self, request):
        city = request.query_params.get("city")
        ai_sports = request.query_params.getlist("ai_sports")
        geoapify_sports = request.query_params.getlist("geoapify_sports") or list(DEFAULT_SPORTS)
        min_confidence = request.query_params.get("min_confidence", "low")
        include_unknown = request.query_params.get("include_unknown", "false").lower() == "true"
        try:
            max_results = int(request.query_params.get("max_results", 50))
        except ValueError:
            return Response({"detail": "max_results must be an integer."}, status=422)

        if not city:
            return Response({"detail": "'city' is required."}, status=422)
        if not ai_sports:
            return Response({"detail": "'ai_sports' is required."}, status=422)

        api_key, err = get_api_key()
        if err:
            return err

        groq_key, err = get_groq_key()
        if err:
            return err

        sport_err = validate_sports(geoapify_sports)
        if sport_err:
            return sport_err

        from ai_categorizer import categorize_venues, filter_by_ai_sport

        venues = scrape_geoapify_venues(
            city=city,
            api_key=api_key,
            sports=geoapify_sports,
            max_results=max_results,
            page_size=20,
            timeout=20,
            delay=0.3,
        )
        venues = categorize_venues(venues, groq_api_key=groq_key)
        venues = filter_by_ai_sport(
            venues,
            sports=ai_sports,
            include_unknown=include_unknown,
            min_confidence=min_confidence,
        )

        return Response({
            "city": city,
            "total": len(venues),
            "venues": serialize_venues(venues),
        })


# ---------------------------------------------------------------------------
# GET /venues/<place_id>?city=...&sport=...
# ---------------------------------------------------------------------------

class VenueDetailView(APIView):
    def get(self, request, place_id: str):
        city = request.query_params.get("city")
        sport = request.query_params.get("sport")

        if not city:
            return Response({"detail": "'city' query param is required."}, status=422)

        api_key, err = get_api_key()
        if err:
            return err

        sports = [sport] if sport and sport in SPORT_CATEGORY_MAP else list(DEFAULT_SPORTS)

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
            return Response(
                {"detail": f"Venue '{place_id}' not found in {city}."},
                status=status.HTTP_404_NOT_FOUND,
            )

        return Response(VenueOutSerializer.from_venue(match))
