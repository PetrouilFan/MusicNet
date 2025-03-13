import hashlib
import os
import json
import sqlite3
import time
import logging
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import yt_dlp
import feature_extract

# Directory paths
LIBRARY_PATH = "library"
FEATURES_PATH = os.path.join(LIBRARY_PATH, "features")
MP3_PATH = os.path.join(LIBRARY_PATH, "mp3")
DB_PATH = os.path.join(LIBRARY_PATH, "musicnet.db")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(LIBRARY_PATH, 'library.log'))
    ]
)
logger = logging.getLogger('musicnet_library')

class Song:
    def __init__(self, title, artist):
        self.title = title
        self.artist = artist
        self.path_of_mp3 = None
        self.features = []
        self.ytb_id = None
        self.playlists = []
        self.id = hashlib.md5(f"{title}-{artist}".encode()).hexdigest()
        self.album = None
        self.release_date = None
        self.genre = None
        self.duration = None
        self.spotify_id = None
        self.last_updated = time.time()
        self._is_modified = True  # Track if this song has been modified since last save
    
    def __str__(self):
        return f"{self.title} by {self.artist}"
    
    def to_dict(self):
        """Convert Song object to dictionary for storage"""
        return {
            "id": self.id,
            "title": self.title,
            "artist": self.artist,
            "path_of_mp3": self.path_of_mp3,
            "features": self.features,
            "ytb_id": self.ytb_id,
            "playlists": self.playlists,
            "album": self.album,
            "release_date": self.release_date,
            "genre": self.genre,
            "duration": self.duration,
            "spotify_id": self.spotify_id,
            "last_updated": self.last_updated,
        }
    
    @classmethod
    def from_dict(cls, data):
        """Create Song object from dictionary data"""
        song = cls(data["title"], data["artist"])
        song.path_of_mp3 = data.get("path_of_mp3")
        song.features = data.get("features", [])
        song.ytb_id = data.get("ytb_id")
        song.playlists = data.get("playlists", [])
        song.id = data.get("id", song.id)  # Use existing ID if provided
        song.album = data.get("album")
        song.release_date = data.get("release_date")
        song.genre = data.get("genre")
        song.duration = data.get("duration")
        song.spotify_id = data.get("spotify_id")
        song.last_updated = data.get("last_updated", time.time())
        song._is_modified = False  # Reset modification flag
        return song
    
class Playlist:
    def __init__(self, name, spotify_link=None):
        self.name = name
        self.songs = []
        self.spotify_link = spotify_link
        self.description = None
        self.last_updated = time.time()
        self._is_modified = True  # Track if this playlist has been modified since last save
    
    def get_songs(self, library=None):
        """
        Get songs from Spotify link if available.
        If library is provided, the songs will be added to the library as well.
        """
        if not self.spotify_link:
            return self.songs
            
        try:
            sp = spotipy.Spotify(client_credentials_manager=SpotifyClientCredentials())
            playlist = sp.playlist(self.spotify_link)
            
            # Update playlist metadata
            self.description = playlist.get('description')
            self._is_modified = True
            
            # Process tracks
            tracks = []
            results = playlist['tracks']
            tracks.extend(results['items'])
            
            # Handle pagination if needed
            while results['next']:
                results = sp.next(results)
                tracks.extend(results['items'])
            
            song_ids = set()  # To avoid duplicates
            for track in tracks:
                if not track['track']:  # Skip null tracks
                    continue
                    
                song_title = track['track']['name']
                song_artist = track['track']['artists'][0]['name']
                
                # Create Song object
                song = Song(song_title, song_artist)
                
                # Add Spotify metadata
                song.spotify_id = track['track']['id']
                song.album = track['track']['album']['name']
                song.release_date = track['track']['album'].get('release_date')
                song.duration = track['track']['duration_ms'] / 1000.0  # Convert to seconds
                
                # Add to playlist if not already present
                if song.id not in song_ids:
                    song_ids.add(song.id)
                    
                    # Check if song already exists in playlist
                    existing_song = next((s for s in self.songs if s.id == song.id), None)
                    if existing_song:
                        # Update existing song with new metadata
                        if not existing_song.album:
                            existing_song.album = song.album
                        if not existing_song.release_date:
                            existing_song.release_date = song.release_date
                        if not existing_song.duration:
                            existing_song.duration = song.duration
                        if not existing_song.spotify_id:
                            existing_song.spotify_id = song.spotify_id
                        existing_song._is_modified = True
                    else:
                        # Add to playlist
                        song.playlists.append(self.name)
                        self.songs.append(song)
                        
                        # Add to library if provided
                        if library:
                            existing_lib_song = library.get_song_by_id(song.id)
                            if existing_lib_song:
                                # Update existing library song
                                if self.name not in existing_lib_song.playlists:
                                    existing_lib_song.playlists.append(self.name)
                                    existing_lib_song._is_modified = True
                            else:
                                # Add new song to library
                                library.songs.append(song)
            
            self._is_modified = True
            logger.info(f"Retrieved {len(self.songs)} songs from playlist '{self.name}'")
            
        except Exception as e:
            logger.error(f"Error retrieving songs from Spotify playlist '{self.name}': {e}")
        
        return self.songs
    
    def __str__(self):
        return self.name
    
    def to_dict(self):
        """Convert Playlist object to dictionary for storage"""
        return {
            "name": self.name,
            "spotify_link": self.spotify_link,
            "description": self.description,
            "songs": [song.id for song in self.songs],  # Store song IDs instead of song objects
            "last_updated": self.last_updated,
        }
    
class Library:
    def __init__(self):
        self.playlists = []
        self.songs = []
        self.db_conn = None
        self.db_cursor = None
        self._last_auto_save = time.time()
        self._auto_save_interval = 300  # 5 minutes
    
    def _init_database(self):
        """Initialize the SQLite database and create tables if they don't exist"""
        os.makedirs(LIBRARY_PATH, exist_ok=True)
        
        # Connect to SQLite database
        self.db_conn = sqlite3.connect(DB_PATH)
        self.db_cursor = self.db_conn.cursor()
        
        # Enable foreign keys
        self.db_cursor.execute("PRAGMA foreign_keys = ON")
        
        # Create tables if they don't exist
        self.db_cursor.execute('''
            CREATE TABLE IF NOT EXISTS songs (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                artist TEXT NOT NULL,
                path_of_mp3 TEXT,
                ytb_id TEXT,
                album TEXT,
                release_date TEXT,
                genre TEXT,
                duration REAL,
                spotify_id TEXT,
                last_updated REAL,
                checksum TEXT
            )
        ''')
        
        self.db_cursor.execute('''
            CREATE TABLE IF NOT EXISTS features (
                song_id TEXT PRIMARY KEY,
                feature_data TEXT NOT NULL,
                last_updated REAL,
                FOREIGN KEY (song_id) REFERENCES songs(id) ON DELETE CASCADE
            )
        ''')
        
        self.db_cursor.execute('''
            CREATE TABLE IF NOT EXISTS playlists (
                name TEXT PRIMARY KEY,
                spotify_link TEXT,
                description TEXT,
                last_updated REAL
            )
        ''')
        
        self.db_cursor.execute('''
            CREATE TABLE IF NOT EXISTS playlist_songs (
                playlist_name TEXT,
                song_id TEXT,
                PRIMARY KEY (playlist_name, song_id),
                FOREIGN KEY (playlist_name) REFERENCES playlists(name) ON DELETE CASCADE,
                FOREIGN KEY (song_id) REFERENCES songs(id) ON DELETE CASCADE
            )
        ''')
        
        # Create indices for better performance
        self.db_cursor.execute("CREATE INDEX IF NOT EXISTS idx_song_title ON songs (title)")
        self.db_cursor.execute("CREATE INDEX IF NOT EXISTS idx_song_artist ON songs (artist)")
        
        # Commit the changes
        self.db_conn.commit()
        
        logger.info("Database initialized")
    
    def add_playlist(self, playlist):
        """Add a playlist to the library"""
        # Check if playlist already exists
        existing = self.get_playlist_by_name(playlist.name)
        if existing:
            logger.warning(f"Playlist '{playlist.name}' already exists in the library")
            return existing
            
        self.playlists.append(playlist)
        playlist._is_modified = True
        logger.info(f"Added playlist '{playlist.name}' to the library")
        return playlist
    
    def get_playlist_by_name(self, name):
        """Get a playlist by name"""
        for playlist in self.playlists:
            if playlist.name == name:
                return playlist
        return None
    
    def get_song_by_id(self, song_id):
        """Get a song by ID"""
        for song in self.songs:
            if song.id == song_id:
                return song
        return None
    
    def migrate_from_json(self):
        """Migrate data from the old JSON format to the new SQLite database"""
        json_path = os.path.join(LIBRARY_PATH, "library.json")
        if not os.path.exists(json_path):
            logger.info("No JSON library file found to migrate")
            return False
            
        try:
            with open(json_path, "r") as f:
                data = json.load(f)
                
            # First load all songs
            song_dict = {}  # Dictionary to map song IDs to song objects
            for song_data in data.get("songs", []):
                song = Song(song_data["title"], song_data["artist"])
                song.path_of_mp3 = song_data.get("path_of_mp3")
                song.features = song_data.get("features", [])
                song.ytb_id = song_data.get("ytb_id")
                song.playlists = song_data.get("playlists", [])
                song.id = song_data.get("id", song.id)
                self.songs.append(song)
                song_dict[song.id] = song
                
            # Now load playlists and connect to song objects
            for playlist_data in data.get("playlists", []):
                playlist = Playlist(playlist_data["name"], None)  # Initialize with no spotify_link
                # Replace song IDs with actual song objects
                playlist.songs = [song_dict.get(song_id) for song_id in playlist_data.get("songs", []) if song_dict.get(song_id)]
                self.playlists.append(playlist)
                
            logger.info(f"Migrated {len(self.songs)} songs and {len(self.playlists)} playlists from JSON")
            
            # Rename the old JSON file as backup
            backup_path = json_path + ".backup"
            os.rename(json_path, backup_path)
            logger.info(f"Renamed old JSON file to {backup_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error migrating from JSON: {e}")
            return False
    
    def save(self, force=False):
        """Save library to SQLite database"""
        current_time = time.time()
        
        # Check if we need to save based on interval (unless force=True)
        if not force and (current_time - self._last_auto_save < self._auto_save_interval):
            # Not enough time has passed since last auto-save
            return False
            
        # Initialize database if not already done
        if self.db_conn is None:
            self._init_database()
            
        modified_songs = 0
        modified_playlists = 0
        
        try:
            # Start a transaction
            self.db_conn.execute("BEGIN TRANSACTION")
            
            # Save songs
            for song in self.songs:
                if hasattr(song, '_is_modified') and song._is_modified:
                    # Check if song exists
                    self.db_cursor.execute("SELECT id FROM songs WHERE id = ?", (song.id,))
                    exists = self.db_cursor.fetchone()
                    
                    song.last_updated = current_time
                    
                    if exists:
                        # Update existing song
                        self.db_cursor.execute('''
                            UPDATE songs SET
                                title = ?,
                                artist = ?,
                                path_of_mp3 = ?,
                                ytb_id = ?,
                                album = ?,
                                release_date = ?,
                                genre = ?,
                                duration = ?,
                                spotify_id = ?,
                                last_updated = ?
                            WHERE id = ?
                        ''', (
                            song.title,
                            song.artist,
                            song.path_of_mp3,
                            song.ytb_id,
                            song.album,
                            song.release_date,
                            song.genre,
                            song.duration,
                            song.spotify_id,
                            song.last_updated,
                            song.id
                        ))
                    else:
                        # Insert new song
                        self.db_cursor.execute('''
                            INSERT INTO songs
                                (id, title, artist, path_of_mp3, ytb_id, album, release_date, 
                                genre, duration, spotify_id, last_updated)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ''', (
                            song.id,
                            song.title,
                            song.artist,
                            song.path_of_mp3,
                            song.ytb_id,
                            song.album,
                            song.release_date,
                            song.genre,
                            song.duration,
                            song.spotify_id,
                            song.last_updated
                        ))
                    
                    # Save features
                    if song.features:
                        # Check if features exist
                        self.db_cursor.execute("SELECT song_id FROM features WHERE song_id = ?", (song.id,))
                        features_exist = self.db_cursor.fetchone()
                        
                        if features_exist:
                            # Update existing features
                            self.db_cursor.execute('''
                                UPDATE features SET
                                    feature_data = ?,
                                    last_updated = ?
                                WHERE song_id = ?
                            ''', (
                                json.dumps(song.features),
                                current_time,
                                song.id
                            ))
                        else:
                            # Insert new features
                            self.db_cursor.execute('''
                                INSERT INTO features (song_id, feature_data, last_updated)
                                VALUES (?, ?, ?)
                            ''', (
                                song.id,
                                json.dumps(song.features),
                                current_time
                            ))
                    
                    song._is_modified = False
                    modified_songs += 1
            
            # Save playlists
            for playlist in self.playlists:
                if hasattr(playlist, '_is_modified') and playlist._is_modified:
                    # Check if playlist exists
                    self.db_cursor.execute("SELECT name FROM playlists WHERE name = ?", (playlist.name,))
                    exists = self.db_cursor.fetchone()
                    
                    playlist.last_updated = current_time
                    
                    if exists:
                        # Update existing playlist
                        self.db_cursor.execute('''
                            UPDATE playlists SET
                                spotify_link = ?,
                                description = ?,
                                last_updated = ?
                            WHERE name = ?
                        ''', (
                            playlist.spotify_link,
                            playlist.description,
                            playlist.last_updated,
                            playlist.name
                        ))
                    else:
                        # Insert new playlist
                        self.db_cursor.execute('''
                            INSERT INTO playlists (name, spotify_link, description, last_updated)
                            VALUES (?, ?, ?, ?)
                        ''', (
                            playlist.name,
                            playlist.spotify_link,
                            playlist.description,
                            playlist.last_updated
                        ))
                    
                    # Clear existing playlist-song mappings
                    self.db_cursor.execute("DELETE FROM playlist_songs WHERE playlist_name = ?", (playlist.name,))
                    
                    # Insert playlist-song mappings
                    for song in playlist.songs:
                        self.db_cursor.execute('''
                            INSERT INTO playlist_songs (playlist_name, song_id)
                            VALUES (?, ?)
                        ''', (playlist.name, song.id))
                    
                    playlist._is_modified = False
                    modified_playlists += 1
            
            # Commit the transaction
            self.db_conn.commit()
            self._last_auto_save = current_time
            
            logger.info(f"Saved library: {modified_songs} songs and {modified_playlists} playlists modified")
            return True
            
        except Exception as e:
            # Rollback in case of error
            self.db_conn.rollback()
            logger.error(f"Error saving library: {e}")
            return False
    
    def load(self):
        """Load library from SQLite database or migrate from JSON if necessary"""
        os.makedirs(LIBRARY_PATH, exist_ok=True)
        os.makedirs(FEATURES_PATH, exist_ok=True)
        os.makedirs(MP3_PATH, exist_ok=True)
        
        # Initialize database if not already done
        if self.db_conn is None:
            self._init_database()
        
        # Clear current data
        self.songs = []
        self.playlists = []
        
        # Check if database has data
        self.db_cursor.execute("SELECT COUNT(*) FROM songs")
        song_count = self.db_cursor.fetchone()[0]
        
        if song_count == 0:
            # Try to migrate from JSON if no songs in database
            migrated = self.migrate_from_json()
            if migrated:
                # Save migrated data to database
                self.save(force=True)
                return True
        
        try:
            # Load songs
            self.db_cursor.execute('''
                SELECT id, title, artist, path_of_mp3, ytb_id, album, release_date, 
                       genre, duration, spotify_id, last_updated
                FROM songs
            ''')
            song_rows = self.db_cursor.fetchall()
            
            song_dict = {}  # Dictionary to map song IDs to song objects
            for row in song_rows:
                song = Song(row[1], row[2])  # title, artist
                song.id = row[0]
                song.path_of_mp3 = row[3]
                song.ytb_id = row[4]
                song.album = row[5]
                song.release_date = row[6]
                song.genre = row[7]
                song.duration = row[8]
                song.spotify_id = row[9]
                song.last_updated = row[10]
                song._is_modified = False
                
                self.songs.append(song)
                song_dict[song.id] = song
            
            # Load features
            self.db_cursor.execute('SELECT song_id, feature_data FROM features')
            feature_rows = self.db_cursor.fetchall()
            
            for row in feature_rows:
                song_id = row[0]
                feature_data = json.loads(row[1])
                
                if song_id in song_dict:
                    song_dict[song_id].features = feature_data
            
            # Load playlists
            self.db_cursor.execute('SELECT name, spotify_link, description, last_updated FROM playlists')
            playlist_rows = self.db_cursor.fetchall()
            
            for row in playlist_rows:
                playlist = Playlist(row[0], row[1])  # name, spotify_link
                playlist.description = row[2]
                playlist.last_updated = row[3]
                playlist._is_modified = False
                self.playlists.append(playlist)
            
            # Load playlist-song mappings
            for playlist in self.playlists:
                self.db_cursor.execute('''
                    SELECT song_id FROM playlist_songs WHERE playlist_name = ?
                ''', (playlist.name,))
                
                song_id_rows = self.db_cursor.fetchall()
                for row in song_id_rows:
                    song_id = row[0]
                    if song_id in song_dict:
                        playlist.songs.append(song_dict[song_id])
                        
                        # Update song's playlists list
                        if playlist.name not in song_dict[song_id].playlists:
                            song_dict[song_id].playlists.append(playlist.name)
            
            logger.info(f"Loaded library: {len(self.songs)} songs and {len(self.playlists)} playlists")
            return True
            
        except Exception as e:
            logger.error(f"Error loading library: {e}")
            return False
    
    def download_all(self):
        """Download all songs in all playlists"""
        total_songs = sum(len(playlist.songs) for playlist in self.playlists)
        downloaded = 0
        skipped = 0
        failed = 0
        
        logger.info(f"Starting download of {total_songs} songs")
        
        for playlist in self.playlists:
            for song in playlist.songs:
                result = self.download_song(song)
                if result == 1:
                    downloaded += 1
                elif result == 0:
                    skipped += 1
                else:
                    failed += 1
                    
                # Auto-save periodically
                if (downloaded + skipped + failed) % 10 == 0:
                    self.save()
        
        # Final save
        self.save(force=True)
        
        logger.info(f"Download complete: {downloaded} downloaded, {skipped} skipped, {failed} failed")
        return downloaded, skipped, failed

    def download_song(self, song):
        """
        Download a song from YouTube.
        Returns: 1 for downloaded, 0 for skipped, -1 for failed
        """
        if song.path_of_mp3 is not None and os.path.exists(song.path_of_mp3):
            logger.info(f"MP3 already exists for '{song.title}' by {song.artist}, skipping download")
            return 0
            
        # Define expected file path before download
        expected_path = os.path.join(MP3_PATH, f"{song.id}.mp3")
        
        # Check if file already exists
        if os.path.exists(expected_path):
            logger.info(f"MP3 already exists for '{song.title}' by {song.artist}, skipping download")
            song.path_of_mp3 = expected_path
            song._is_modified = True
            return 0
                
        query = f"{song.title} {song.artist}"
        ydl_opts = {
            'format': 'bestaudio/best',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }],
            'outtmpl': os.path.join(MP3_PATH, f"{song.id}.%(ext)s")
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            try:
                # First extract info without downloading to check duration
                info = ydl.extract_info(f"ytsearch1:{query}", download=False)
                if 'entries' in info:
                    video = info['entries'][0]
                else:
                    video = info
                
                # Check if the video is too long (>10 minutes)
                duration = video.get('duration', 0)
                if duration > 600:  # 600 seconds = 10 minutes
                    logger.warning(f"Skipped '{song.title}' by {song.artist}: Too long ({duration/60:.1f} minutes)")
                    return -1
                
                # Video is within acceptable length, download it
                song.ytb_id = video['id']
                ydl.download([f"https://www.youtube.com/watch?v={video['id']}"])
                
                # Get the file path of the downloaded file
                file_path = ydl.prepare_filename(video)
                # Convert the extension to mp3 since that's what we're extracting to
                song.path_of_mp3 = os.path.splitext(file_path)[0] + '.mp3'
                song._is_modified = True
                
                logger.info(f"Downloaded '{song.title}' by {song.artist}")
                return 1
                
            except Exception as e:
                logger.error(f"Error downloading '{song.title}' by {song.artist}: {e}")
                return -1

    def extract_all(self, parallel=True, max_processes=None):
        """
        Extract features for all songs.
        Args:
            parallel: Whether to use parallel processing
            max_processes: Number of processes to use (defaults to feature_extract.MAX_PROCESSES)
        """
        if parallel:
            # Use parallel feature extraction
            feature_results = feature_extract.parallel_extract_features(
                self, 
                max_processes=max_processes or feature_extract.MAX_PROCESSES
            )
            
            # Update library with extracted features
            feature_extract.update_library_with_features(self, feature_results)
        else:
            # Use sequential feature extraction
            for playlist in self.playlists:
                for song in playlist.songs:
                    if len(song.features) == 0 and song.path_of_mp3 and os.path.exists(song.path_of_mp3):
                        song.features = feature_extract.extract_features(song)
                        song._is_modified = True
        
        # Save the updated library
        self.save(force=True)
    
    def get_songs_without_features(self):
        """Get a list of songs that don't have features extracted"""
        songs_without_features = []
        for song in self.songs:
            if len(song.features) == 0 and song.path_of_mp3 and os.path.exists(song.path_of_mp3):
                songs_without_features.append(song)
        return songs_without_features
    
    def get_songs_without_mp3(self):
        """Get a list of songs that don't have MP3 files"""
        songs_without_mp3 = []
        for song in self.songs:
            if song.path_of_mp3 is None or not os.path.exists(song.path_of_mp3):
                songs_without_mp3.append(song)
        return songs_without_mp3
    
    def find_duplicates(self):
        """Find duplicate songs in the library based on title and artist"""
        song_dict = {}
        duplicates = []
        
        for song in self.songs:
            key = f"{song.title.lower()}-{song.artist.lower()}"
            if key in song_dict:
                duplicates.append((song_dict[key], song))
            else:
                song_dict[key] = song
                
        return duplicates
    
    def cleanup_duplicates(self):
        """Clean up duplicate songs in the library"""
        duplicates = self.find_duplicates()
        removed = 0
        
        for orig, dupe in duplicates:
            # Keep the song with more data
            if len(orig.features) < len(dupe.features):
                orig, dupe = dupe, orig
                
            # Update playlists to use the original song
            for playlist in self.playlists:
                if dupe in playlist.songs:
                    if orig not in playlist.songs:
                        playlist.songs.append(orig)
                    playlist.songs.remove(dupe)
                    playlist._is_modified = True
                    
            # Remove the duplicate from the library
            self.songs.remove(dupe)
            removed += 1
            
        logger.info(f"Removed {removed} duplicate songs")
        
        # Save changes
        if removed > 0:
            self.save(force=True)
            
        return removed
    
    def export_json(self, filepath=None):
        """Export library to JSON file"""
        if filepath is None:
            filepath = os.path.join(LIBRARY_PATH, "library_export.json")
            
        data = {
            "songs": [song.to_dict() for song in self.songs],
            "playlists": [playlist.to_dict() for playlist in self.playlists]
        }
        
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)
            
        logger.info(f"Exported library to {filepath}")
        return filepath
    
    def generate_library_stats(self):
        """Generate statistics about the library"""
        stats = {
            "total_songs": len(self.songs),
            "total_playlists": len(self.playlists),
            "songs_with_features": sum(1 for song in self.songs if song.features),
            "songs_with_mp3": sum(1 for song in self.songs if song.path_of_mp3 and os.path.exists(song.path_of_mp3)),
            "total_duration_hours": sum(song.duration or 0 for song in self.songs) / 3600 if any(song.duration for song in self.songs) else 0,
            "playlists": {}
        }
        
        # Playlist stats
        for playlist in self.playlists:
            stats["playlists"][playlist.name] = {
                "songs_count": len(playlist.songs),
                "songs_with_features": sum(1 for song in playlist.songs if song.features),
                "songs_with_mp3": sum(1 for song in playlist.songs if song.path_of_mp3 and os.path.exists(song.path_of_mp3)),
                "spotify_link": playlist.spotify_link,
            }
        
        return stats
    
    def close(self):
        """Close database connection"""
        if self.db_conn:
            self.db_conn.close()
            self.db_conn = None
            self.db_cursor = None
            logger.info("Database connection closed")
