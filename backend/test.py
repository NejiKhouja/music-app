# test.py - Save this in backend directory
import requests
import json
import sys
import os

# Add current directory to path so we can import Django
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_api():
    print("🎵 Testing Music Recommendation API")
    print("=" * 50)
    
    # First, we need to check if the server is running
    print("1. Checking if server is running...")
    
    try:
        # Try to get the home page
        response = requests.get('http://127.0.0.1:8000/api/', timeout=3)
        if response.status_code == 200:
            print("   ✅ Server is running!")
            print(f"   Message: {response.json().get('message', 'No message')}")
        else:
            print(f"   ⚠️  Server responded with status: {response.status_code}")
    except requests.ConnectionError:
        print("   ❌ Server is NOT running!")
        print("   Please run: python manage.py runserver")
        return False
    
    # Test 2: Get tracks
    print("\n2. Getting track list...")
    try:
        response = requests.get('http://127.0.0.1:8000/api/tracks/', timeout=5)
        if response.status_code == 200:
            data = response.json()
            tracks = data.get('tracks', [])
            print(f"   ✅ Found {len(tracks)} tracks")
            
            if tracks:
                # Show first track
                track = tracks[0]
                print(f"   Sample track: {track['track_name']} by {track['artist_name']}")
                print(f"   ML Index: {track['ml_index']}")
                
                # Test 3: Get recommendations
                print("\n3. Testing recommendations...")
                test_data = {
                    'ml_index': track['ml_index'],
                    'n': 3
                }
                
                response = requests.post(
                    'http://127.0.0.1:8000/api/recommend/',
                    json=test_data,
                    timeout=10
                )
                
                if response.status_code == 200:
                    rec_data = response.json()
                    print(f"   ✅ Recommendations received!")
                    print(f"   For: {rec_data['query_track']['track_name']}")
                    
                    recommendations = rec_data.get('recommendations', [])
                    if recommendations:
                        print(f"   Top {len(recommendations)} recommendations:")
                        for i, rec in enumerate(recommendations, 1):
                            print(f"     {i}. {rec['track_name']} - Similarity: {rec['similarity']:.3f}")
                    else:
                        print("   ⚠️  No recommendations returned")
                else:
                    print(f"   ❌ Recommendation failed: {response.status_code}")
                    print(f"   Response: {response.text}")
            else:
                print("   ⚠️  No tracks in database")
                print("   Run: python manage.py import_tracks \"path/to/metadata.csv\"")
        
        else:
            print(f"   ❌ Failed to get tracks: {response.status_code}")
            
    except Exception as e:
        print(f"   ❌ Error during test: {e}")
    
    print("\n" + "=" * 50)
    print("Test complete!")

def quick_test_without_server():
    """Test without needing the server to run"""
    print("🔧 Quick Database & ML Test (No server needed)")
    print("=" * 50)
    
    try:
        # Setup Django
        os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'backend.settings')
        import django
        django.setup()
        
        from music.models import Track
        from music.services.recommender import recommender
        
        # Test 1: Database
        print("1. Checking database...")
        count = Track.objects.count()
        print(f"   Tracks in database: {count}")
        
        if count > 0:
            track = Track.objects.first()
            print(f"   Sample: ML Index {track.ml_index}, '{track.track_name}'")
            
            # Test 2: ML Model
            print("\n2. Checking ML model...")
            if recommender.loaded:
                print(f"   ✅ ML model loaded - {len(recommender.features)} tracks")
                
                # Test 3: Get recommendations directly
                print("\n3. Testing ML recommendations...")
                recs = recommender.get_recommendations(track.ml_index, n=3)
                
                if recs:
                    print(f"   Found {len(recs)} recommendations:")
                    for i, rec in enumerate(recs, 1):
                        try:
                            rec_track = Track.objects.get(ml_index=rec['ml_index'])
                            similarity = 1 - rec['distance']
                            print(f"     {i}. {rec_track.track_name} - Similarity: {similarity:.3f}")
                        except:
                            print(f"     {i}. ML Index {rec['ml_index']} (not in DB)")
                else:
                    print("   ⚠️  No recommendations from ML model")
            else:
                print("   ❌ ML model not loaded")
                print("   Check your ML files at: C:\\Users\\deadx\\OneDrive\\Desktop\\music-app\\ml")
        
        else:
            print("   ⚠️  Database is empty")
            print("   Run: python manage.py import_tracks \"C:\\Users\\deadx\\OneDrive\\Desktop\\music-app\\ml\\src\\metadata\\tracks_metadata.csv\"")
    
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 50)

if __name__ == "__main__":
    # First try quick test without server
    quick_test_without_server()
    
    print("\n\n" + "=" * 50)
    print("SERVER TEST (requires running server)")
    print("=" * 50)
    print("Run this test AFTER starting server with: python manage.py runserver")
    print("Then in another terminal, run: python test.py")
    
    # Ask if they want to run server test
    response = input("\nDo you want to test with server now? (y/n): ")
    if response.lower() == 'y':
        test_api()