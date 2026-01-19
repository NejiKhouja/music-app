from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json
from music.models import Track
from music.services.recommender import recommender
from django.http import JsonResponse
from music.models import Track
# views.py - Update both functions to include track_id


# Add this to views.py - AFTER your existing views
import pandas as pd
import os

# Add to views.py
from django.db.models import Q
from django.db.models.functions import Lower
from django.core.paginator import Paginator

def search_all(request):
    """Search for both tracks and distinct artists"""
    query = request.GET.get('q', '').strip().lower()
    
    if not query or len(query) < 2:
        return JsonResponse({
            'tracks': [],
            'artists': [],
            'total': 0
        })
    
    # Search tracks (case-insensitive)
    tracks = Track.objects.filter(
        Q(track_name__icontains=query) |
        Q(artist_name__icontains=query)
    ).order_by('track_name').Paginator(tracks, 10).page(1).object_list
    
    # Get distinct artists that match the query
    # This gives us unique artists from tracks that match
    matching_artists = Track.objects.filter(
        artist_name__icontains=query
    ).values_list('artist_name', flat=True).distinct()[:10]
    
    # Prepare track data
    tracks_data = [
        {
            'type': 'song',
            'id': t.track_id or str(t.ml_index),
            'track_id': t.track_id,
            'ml_index': t.ml_index,
            'title': t.track_name,
            'artist': t.artist_name,
            'genre': t.genre,
            'score': calculate_search_score(t, query)
        }
        for t in tracks
    ]
    
    # Prepare artist data (as simple strings or objects)
    artists_data = [
        {
            'type': 'artist',
            'id': artist_name,      
            'name': artist_name,
            'track_count': Track.objects.filter(artist_name=artist_name).count()
        }
        for artist_name in matching_artists
    ]
    
    return JsonResponse({
        'tracks': tracks_data,
        'artists': artists_data,
        'total': len(tracks_data) + len(artists_data)
    })

# Optional: Add scoring to rank results better
def calculate_search_score(self, track, query):
    """Calculate a relevance score for search results"""
    score = 0
    
    # Exact match in track name gets highest score
    if query in track.track_name.lower():
        score += 100
        # Exact match at beginning gets even higher
        if track.track_name.lower().startswith(query):
            score += 50
    
    # Exact match in artist name
    if query in track.artist_name.lower():
        score += 80
        if track.artist_name.lower().startswith(query):
            score += 40
    
    # Partial matches
    if track.track_name.lower().find(query) != -1:
        score += 30
    
    if track.artist_name.lower().find(query) != -1:
        score += 20
    
    return score

def get_audio_features(request, track_id):
    """Get audio features for a track from your ML features"""
    try:
        track = Track.objects.get(track_id=track_id)
        
        # Load your ML features CSV
        features_path = "ml/src/features/features.csv"  # Adjust this path if needed
        if os.path.exists(features_path):
            df_features = pd.read_csv(features_path)
            
            # Find the track in features (track_id column)
            if 'track_id' not in df_features.columns:
                return JsonResponse({'error': 'Features CSV does not have track_id column'}, status=500)
            
            track_features = df_features[df_features['track_id'] == track_id]
            
            if track_features.empty:
                return JsonResponse({'error': 'Features not found for this track'}, status=404)
            
            # Get the first matching row
            features = track_features.iloc[0].to_dict()
            
            # Remove track_id from response
            if 'track_id' in features:
                del features['track_id']
                
            return JsonResponse(features)
        else:
            return JsonResponse({'error': 'Features file not found'}, status=404)
            
    except Track.DoesNotExist:
        return JsonResponse({'error': 'Track not found'}, status=404)
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)

def get_track_with_details(request, track_id):
    """Get track with all details (track info, audio features, similar tracks)"""
    try:
        track = Track.objects.get(track_id=track_id)
        
        # Get track info
        track_data = {
            'ml_index': track.ml_index,
            'track_id': track.track_id,
            'track_name': track.track_name,
            'artist_name': track.artist_name,
            'genre': track.genre
        }
        
        # Get audio features
        audio_features = None
        try:
            features_path = "ml/src/features/features.csv"
            if os.path.exists(features_path):
                df_features = pd.read_csv(features_path)
                if 'track_id' in df_features.columns:
                    track_features = df_features[df_features['track_id'] == track_id]
                    if not track_features.empty:
                        audio_features = track_features.iloc[0].to_dict()
                        if 'track_id' in audio_features:
                            del audio_features['track_id']
        except Exception as e:
            print(f"Error getting audio features: {e}")
        
        # Get similar tracks
        similar_tracks = []
        try:
            recs = recommender.get_recommendations(track.ml_index, 10)
            for rec in recs:
                try:
                    rec_track = Track.objects.get(ml_index=rec['ml_index'])
                    similar_tracks.append({
                        'ml_index': rec_track.ml_index,
                        'track_id': rec_track.track_id,
                        'track_name': rec_track.track_name,
                        'artist_name': rec_track.artist_name,
                        'genre': rec_track.genre,
                        'similarity': 1 - rec['distance']
                    })
                except Track.DoesNotExist:
                    continue
        except Exception as e:
            print(f"Error getting similar tracks: {e}")
        
        return JsonResponse({
            'track': track_data,
            'audio_features': audio_features,
            'similar_tracks': similar_tracks
        })
        
    except Track.DoesNotExist:
        return JsonResponse({'error': 'Track not found'}, status=404)
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)

# views.py - ADD THIS VIEW
def get_tracks_by_artist(request):
    """Get tracks by artist name with more flexible search"""
    artist_name = request.GET.get('artist', '').strip().lower()
    
    if not artist_name:
        return JsonResponse({'tracks': [], 'count': 0})
    
    # Split search terms to handle multiple words
    search_terms = artist_name.split()
    
    # Start with all tracks and filter
    tracks = Track.objects.all()
    
    # Filter by each search term
    for term in search_terms:
        tracks = tracks.filter(artist_name__icontains=term)
    
    tracks = tracks[:50]
    
    data = [
        {
            'ml_index': t.ml_index,
            'track_id': t.track_id,
            'track_name': t.track_name,
            'artist_name': t.artist_name,
            'genre': t.genre
        }
        for t in tracks
    ]
    
    return JsonResponse({'tracks': data, 'count': len(data)})

@csrf_exempt
def get_track_by_id(request, track_id):
    """Get a single track by its track_id"""
    try:
        track = Track.objects.get(track_id=track_id)
        data = {
            'ml_index': track.ml_index,
            'track_id': track.track_id,
            'track_name': track.track_name,
            'artist_name': track.artist_name,
            'genre': track.genre
        }
        return JsonResponse(data)
    except Track.DoesNotExist:
        return JsonResponse({'error': f'Track {track_id} not found'}, status=404)

@csrf_exempt
def recommend_by_track_id(request, track_id):
    """Get recommendations for track by track_id"""
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            n = data.get('n', 5)
            
            # Get track by track_id
            try:
                track = Track.objects.get(track_id=track_id)
            except Track.DoesNotExist:
                return JsonResponse({'error': f'Track {track_id} not found'}, status=404)
            
            # Get recommendations using ml_index
            recs = recommender.get_recommendations(track.ml_index, n)
            
            # Format response
            recommendations = []
            for rec in recs:
                try:
                    rec_track = Track.objects.get(ml_index=rec['ml_index'])
                    recommendations.append({
                        'ml_index': rec_track.ml_index,
                        'track_id': rec_track.track_id,
                        'track_name': rec_track.track_name,
                        'artist_name': rec_track.artist_name,
                        'genre': rec_track.genre,
                        'similarity': 1 - rec['distance']
                    })
                except Track.DoesNotExist:
                    continue
            
            return JsonResponse({
                'query_track': {
                    'ml_index': track.ml_index,
                    'track_id': track.track_id,
                    'track_name': track.track_name,
                    'artist_name': track.artist_name,
                    'genre': track.genre
                },
                'recommendations': recommendations
            })
            
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)
    
    # Also support GET requests
    elif request.method == 'GET':
        n = request.GET.get('n', 5)
        try:
            n = int(n)
        except:
            n = 5
            
        # Same logic as above...
        # (Copy the POST logic here or refactor into a helper function)
        
    return JsonResponse({'error': 'Use POST or GET'}, status=405)




@csrf_exempt
def recommend(request):
    """Get recommendations for track by ML index"""
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            ml_index = data.get('ml_index')
            n = data.get('n', 5)
            
            if ml_index is None:
                return JsonResponse({'error': 'ml_index required'}, status=400)
            
            # Validate input
            try:
                ml_index = int(ml_index)
            except:
                return JsonResponse({'error': 'ml_index must be a number'}, status=400)
            
            # Get the track
            try:
                track = Track.objects.get(ml_index=ml_index)
            except Track.DoesNotExist:
                return JsonResponse({'error': f'Track {ml_index} not found'}, status=404)
            
            # Get recommendations from ML
            recs = recommender.get_recommendations(ml_index, n)
            
            # Format response
            recommendations = []
            for rec in recs:
                try:
                    rec_track = Track.objects.get(ml_index=rec['ml_index'])
                    recommendations.append({
                        'ml_index': rec_track.ml_index,
                        'track_id': rec_track.track_id,  # ADD THIS
                        'track_name': rec_track.track_name,
                        'artist_name': rec_track.artist_name,
                        'genre': rec_track.genre,  # ADD THIS
                        'similarity': 1 - rec['distance']
                    })
                except:
                    continue
            
            return JsonResponse({
                'query_track': {
                    'ml_index': track.ml_index,
                    'track_id': track.track_id,  # ADD THIS
                    'track_name': track.track_name,
                    'artist_name': track.artist_name,
                    'genre': track.genre  # ADD THIS
                },
                'recommendations': recommendations
            })
            
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)
    
    return JsonResponse({'error': 'Use POST'}, status=405)


def get_tracks(request):
    ml_index = request.GET.get('ml_index')
    if ml_index is not None:
        try:
            track = Track.objects.get(ml_index=int(ml_index))
            data = [{
                'ml_index': track.ml_index,
                'track_id': track.track_id,  # ADD THIS
                'track_name': track.track_name,
                'artist_name': track.artist_name,
                'genre': track.genre
            }]
            return JsonResponse({'tracks': data, 'count': 1})
        except Track.DoesNotExist:
            return JsonResponse({'tracks': [], 'count': 0})

    # fallback: return first 50 tracks
    tracks = Track.objects.all()[:50]
    data = [
        {
            'ml_index': t.ml_index,
            'track_id': t.track_id,  # ADD THIS
            'track_name': t.track_name,
            'artist_name': t.artist_name,
            'genre': t.genre
        }
        for t in tracks
    ]
    return JsonResponse({'tracks': data, 'count': len(data)})
def home(request):
    return JsonResponse({
        'message': 'Music Recommendation API',
        'endpoints': {
            'GET /': 'This message',
            'GET /tracks/': 'Get track list',
            'POST /recommend/': 'Get recommendations (send {"ml_index": 0, "n": 5})'
        }
    })