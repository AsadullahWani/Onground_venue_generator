from rest_framework import serializers
from .geoapify_sports_venues import DEFAULT_SPORTS, SPORT_CATEGORY_MAP


class VenueOutSerializer(serializers.Serializer):
    place_id             = serializers.CharField()
    name                 = serializers.CharField()
    city                 = serializers.CharField()
    formatted_address    = serializers.CharField()
    short_address        = serializers.CharField()
    latitude             = serializers.FloatField(allow_null=True)
    longitude            = serializers.FloatField(allow_null=True)
    categories           = serializers.CharField()
    sport_type           = serializers.CharField()
    search_category      = serializers.CharField()
    website              = serializers.CharField()
    phone                = serializers.CharField()
    opening_hours        = serializers.CharField()
    country              = serializers.CharField()
    state                = serializers.CharField()
    postcode             = serializers.CharField()
    street               = serializers.CharField()
    distance             = serializers.FloatField(allow_null=True)
    ai_summary           = serializers.CharField()
    ai_sport_type        = serializers.CharField()
    ai_confidence        = serializers.CharField()

    @staticmethod
    def from_venue(venue) -> dict:
        """Convert a Venue dataclass/object to a serializable dict."""
        fields = [
            "place_id", "name", "city", "formatted_address", "short_address",
            "latitude", "longitude", "categories", "sport_type", "search_category",
            "website", "phone", "opening_hours", "country", "state", "postcode",
            "street", "distance", "ai_summary", "ai_sport_type", "ai_confidence",
        ]
        return {f: getattr(venue, f) for f in fields}


class VenueListResponseSerializer(serializers.Serializer):
    city   = serializers.CharField()
    total  = serializers.IntegerField()
    venues = VenueOutSerializer(many=True)


class CityGeocodeOutSerializer(serializers.Serializer):
    city      = serializers.CharField()
    place_id  = serializers.CharField(allow_null=True)
    latitude  = serializers.FloatField(allow_null=True)
    longitude = serializers.FloatField(allow_null=True)


class SportCategorySerializer(serializers.Serializer):
    name               = serializers.CharField()
    geoapify_category  = serializers.CharField()


class SportListResponseSerializer(serializers.Serializer):
    total  = serializers.IntegerField()
    sports = SportCategorySerializer(many=True)


class VenueSearchRequestSerializer(serializers.Serializer):
    city = serializers.CharField()
    sports = serializers.ListField(
        child=serializers.CharField(),
        default=list(DEFAULT_SPORTS),
    )
    max_results = serializers.IntegerField(default=50, min_value=1, max_value=500)
    page_size   = serializers.IntegerField(default=20, min_value=1, max_value=500)
    include_ai_summary = serializers.BooleanField(default=False)
    ai_categorize      = serializers.BooleanField(default=False)
    ai_filter_sports   = serializers.ListField(child=serializers.CharField(), default=list)
    ai_min_confidence  = serializers.ChoiceField(
        choices=["high", "medium", "low"], default="low"
    )
    ai_include_unknown = serializers.BooleanField(default=False)

    def validate_sports(self, value):
        invalid = [s for s in value if s not in SPORT_CATEGORY_MAP]
        if invalid:
            raise serializers.ValidationError(
                f"Unknown sport(s): {invalid}. Valid: {list(SPORT_CATEGORY_MAP.keys())}"
            )
        return value
