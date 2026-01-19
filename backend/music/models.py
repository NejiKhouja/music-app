# models.py
from django.db import models

class Track(models.Model):
    # ML uses row numbers 0,1,2... as IDs
    ml_index = models.IntegerField(primary_key=True)
    # Add track_id (16 chars from your MD5 hash)
    track_id = models.CharField(max_length=32, blank=True, null=True, unique=True)
    track_name = models.CharField(max_length=255)
    artist_name = models.CharField(max_length=255)
    genre = models.CharField(max_length=100)
    
    def __str__(self):
        return f"{self.ml_index}: {self.track_name} - {self.artist_name}"

    class Meta:
        # Optional: Add database index for faster lookups
        indexes = [
            models.Index(fields=['track_id']),
            models.Index(fields=['track_name', 'artist_name']),
        ]