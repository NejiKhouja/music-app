# urls.py
from django.urls import path
from . import views

# urls.py - Add these patterns
urlpatterns = [
    path('', views.home, name='home'),
    path('tracks/', views.get_tracks, name='get_tracks'),
    path('recommend/', views.recommend, name='recommend'),
    path('track/<str:track_id>/', views.get_track_by_id, name='get_track_by_id'),
    path('track/<str:track_id>/recommend/', views.recommend_by_track_id, name='recommend_by_track_id'),
    # ADD THESE NEW ENDPOINTS:
    path('track/<str:track_id>/audio-features/', views.get_audio_features, name='get_audio_features'),
    path('track/<str:track_id>/with-details/', views.get_track_with_details, name='get_track_with_details'),
    # urls.py
    path('tracks/by-artist/', views.get_tracks_by_artist, name='get_tracks_by_artist'),
    path('search/', views.search_all, name='search_all'),

]