# -----------------------------------------------------------------------------
# MusicNet Settings Example
# -----------------------------------------------------------------------------
# Instructions:
# 1. Rename this file to settings.py
# 2. Fill in your Spotify API credentials
# 3. Add your playlist URLs
# -----------------------------------------------------------------------------

# Your Spotify API credentials - Get these from Spotify Developer Dashboard
# https://developer.spotify.com/dashboard/
SPOTIFY_CLIENT_ID = "your_client_id_here"
SPOTIFY_CLIENT_SECRET = "your_client_secret_here"

# List of Spotify playlist URLs to process
# Format: Each URL should be in quotation marks and separated by commas
# Example: A list of playlist URLs from your Spotify account
SPOTIFY_PLAYLIST_URLs = [
    "https://open.spotify.com/playlist/playlist_id_1?si=unique_identifier", 
    "https://open.spotify.com/playlist/playlist_id_2?si=unique_identifier",
    # Add more playlists as needed
]