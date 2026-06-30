from django.urls import path
from . import views

urlpatterns = [
    path("health",           views.HealthView.as_view(),       name="health"),
    path("sports",           views.SportListView.as_view(),    name="sports"),
    path("geocode",          views.GeocodeView.as_view(),      name="geocode"),
    path("venues",           views.VenueListView.as_view(),    name="venues-list"),
    path("venues/search",    views.VenueSearchView.as_view(),  name="venues-search"),
    path("venues/by-sport",  views.VenueBySportView.as_view(), name="venues-by-sport"),
    path("venues/<str:place_id>", views.VenueDetailView.as_view(), name="venue-detail"),
]
