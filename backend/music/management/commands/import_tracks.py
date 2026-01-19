# management/commands/import_tracks.py
import csv
import os
import hashlib
from django.core.management.base import BaseCommand
from django.db import transaction
from music.models import Track

class Command(BaseCommand):
    help = "Import tracks with ML indices and track IDs"
    
    def add_arguments(self, parser):
        parser.add_argument('csv_file', type=str, help='Path to metadata CSV')
    
    def generate_track_id(self, track_name: str, artist_name: str) -> str:
        """Generate consistent track_id (same as ML pipeline)"""
        unique_string = f"{track_name}_{artist_name}".lower().strip()
        return hashlib.md5(unique_string.encode()).hexdigest()[:16]
    
    def handle(self, *args, **options):
        csv_file = options['csv_file']
        
        if not os.path.exists(csv_file):
            self.stdout.write(self.style.ERROR(f"File not found: {csv_file}"))
            return
        
        # Clear existing data (optional)
        Track.objects.all().delete()
        self.stdout.write("Cleared existing tracks")
        
        tracks = []
        imported = 0
        errors = 0
        
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            
            for i, row in enumerate(reader):
                try:
                    # Get data - adjust column names based on your CSV
                    track_name = row.get('track_name', '').strip()
                    artist_name = row.get('artist_name', '').strip()
                    genre = row.get('genre', '').strip()
                    
                    # Check if CSV has track_id column
                    track_id = row.get('track_id', '').strip()
                    
                    # Generate track_id if not in CSV
                    if not track_id and track_name and artist_name:
                        track_id = self.generate_track_id(track_name, artist_name)
                    
                    # Skip if no track name or artist
                    if not track_name or not artist_name:
                        self.stdout.write(f"⚠️ Skipping row {i}: Missing track_name or artist_name")
                        errors += 1
                        continue
                    
                    # Create Track object
                    tracks.append(Track(
                        ml_index=i,  # Use CSV row index
                        track_id=track_id if track_id else None,
                        track_name=track_name,
                        artist_name=artist_name,
                        genre=genre
                    ))
                    imported += 1
                    
                    # Bulk insert every 1000 records
                    if len(tracks) >= 1000:
                        with transaction.atomic():
                            Track.objects.bulk_create(tracks, ignore_conflicts=True)
                        self.stdout.write(f"✅ Imported {imported} tracks...")
                        tracks = []
                        
                except Exception as e:
                    errors += 1
                    if errors <= 5:  # Show first 5 errors
                        self.stdout.write(f"❌ Error in row {i}: {e}")
        
        # Insert remaining records
        if tracks:
            with transaction.atomic():
                Track.objects.bulk_create(tracks, ignore_conflicts=True)
        
        self.stdout.write(self.style.SUCCESS(
            f"\n🎉 Import complete!\n"
            f"   Imported: {imported} tracks\n"
            f"   Errors: {errors}\n"
            f"   Total in DB: {Track.objects.count()}"
        ))
        
        # Show sample
        self.stdout.write("\n📋 Sample of imported tracks:")
        for track in Track.objects.all()[:5]:
            self.stdout.write(f"   {track.ml_index}: {track.track_name} - {track.artist_name}")
            self.stdout.write(f"     Track ID: {track.track_id}")