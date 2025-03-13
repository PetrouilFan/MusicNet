import os
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import logging
import time
import feature_extract

import musicnet_library
from settings import SPOTIFY_CLIENT_ID, SPOTIFY_CLIENT_SECRET, SPOTIFY_PLAYLIST_URLs

# Configuration constants
# ===========================
# Feature extraction settings
USE_PARALLEL_PROCESSING = True
NUM_PROCESSES = 4  # Number of parallel processes for feature extraction
FORCE_EXTRACTION = False  # Whether to force re-extraction of features
ANALYSIS_DURATION = 60  # Duration in seconds to analyze for each song
SAMPLE_RATE = 22050  # Audio sample rate for analysis

# Processing flags
SKIP_DOWNLOAD = False  # Set to True to skip downloading MP3 files
SKIP_EXTRACTION = False  # Set to True to skip feature extraction
SKIP_PLOTS = False  # Set to True to skip generating plots
GENERATE_MATRIX_PLOT = False  # Set to True to generate additional feature matrix plot
GENERATE_STATISTICS = True  # Set to True to generate library statistics

# Feature selection for visualization
FEATURE_INDICES = [0, 1, 2]  # BPM, Dominant Frequency, Spectral Centroid

# Visualization paths
VISUALIZATION_3D_PATH = "visualizations/music_features_3d.html"
VISUALIZATION_MATRIX_PATH = "visualizations/feature_matrix.html"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join('library', 'plotter.log'))
    ]
)
logger = logging.getLogger('plotter')

def ensure_library():
    """Load existing library or create a new one with playlists from settings"""
    # Set up Spotify credentials 
    os.environ['SPOTIPY_CLIENT_ID'] = SPOTIFY_CLIENT_ID
    os.environ['SPOTIPY_CLIENT_SECRET'] = SPOTIFY_CLIENT_SECRET
    
    # Create directories if they don't exist
    os.makedirs(musicnet_library.LIBRARY_PATH, exist_ok=True)
    os.makedirs(musicnet_library.FEATURES_PATH, exist_ok=True)
    os.makedirs(musicnet_library.MP3_PATH, exist_ok=True)
    os.makedirs('visualizations', exist_ok=True)
    
    library = musicnet_library.Library()
    
    # Load library from SQLite or migrate from JSON if necessary
    if not library.load():
        logger.error("Failed to load library")
        # Initialize songs attribute if needed
        if not hasattr(library, 'songs'):
            library.songs = []
    
    # Add playlists from settings if not already in library
    for playlist_url in SPOTIFY_PLAYLIST_URLs:
        # Check if this playlist URL already exists in library
        exists = False
        for playlist in library.playlists:
            if hasattr(playlist, 'spotify_link') and playlist.spotify_link == playlist_url:
                exists = True
                logger.info(f"Playlist already exists: {playlist.name}")
                break
                
        if not exists:
            import spotipy
            from spotipy.oauth2 import SpotifyClientCredentials
            
            sp = spotipy.Spotify(client_credentials_manager=SpotifyClientCredentials())
            playlist_data = sp.playlist(playlist_url)
            playlist_name = playlist_data['name']
            
            logger.info(f"Adding new playlist: {playlist_name}")
            new_playlist = musicnet_library.Playlist(playlist_name, playlist_url)
            library.add_playlist(new_playlist)
            
            # Fetch songs for the new playlist
            new_playlist.get_songs(library)
    
    # Save any changes
    library.save()
    return library

def download_songs(library):
    """Download all songs in the library that need downloading"""
    start_time = time.time()
    
    logger.info("Downloading songs...")
    downloaded, skipped, failed = library.download_all()
    logger.info(f"Download summary: {downloaded} downloaded, {skipped} skipped, {failed} failed")
    
    elapsed_time = time.time() - start_time
    logger.info(f"Downloading completed in {elapsed_time:.1f} seconds")
    
    # Save the updated library
    library.save(force=True)
    return library

def extract_features(library, parallel=USE_PARALLEL_PROCESSING, max_processes=NUM_PROCESSES, force=FORCE_EXTRACTION):
    """
    Extract features for songs in the library.
    
    Args:
        library: The music library containing songs
        parallel: Whether to use parallel processing
        max_processes: Number of parallel processes to use
        force: Whether to force re-extraction of features even if they already exist
    """
    start_time = time.time()
    
    # Check if there are songs needing feature extraction
    songs_without_features = library.get_songs_without_features()
    
    if not songs_without_features and not force:
        logger.info("No songs require feature extraction - using cached features")
        return library
    
    # If force is True, we need to consider all songs with MP3s
    if force:
        songs_to_process = []
        for song in library.songs:
            if song.path_of_mp3 and os.path.exists(song.path_of_mp3):
                songs_to_process.append(song)
        logger.info(f"Force re-extracting features for all {len(songs_to_process)} songs")
    else:
        songs_to_process = songs_without_features
        logger.info(f"Extracting features for {len(songs_to_process)} songs")
    
    if songs_to_process:
        if parallel:
            logger.info(f"Using parallel extraction with {max_processes} processes")
            library.extract_all(parallel=True, max_processes=max_processes)
        else:
            logger.info("Using sequential extraction")
            library.extract_all(parallel=False)
        
        # Save the updated library
        library.save(force=True)
    
    elapsed_time = time.time() - start_time
    logger.info(f"Feature extraction completed in {elapsed_time:.1f} seconds")
    
    return library

def plot_features_3d(library, feature_indices=FEATURE_INDICES, output_file=VISUALIZATION_3D_PATH):
    """
    Create 3D visualization of song features using plotly.
    
    Args:
        library: The music library containing songs and playlists
        feature_indices: List of 3 indices to use for x, y, z axes
        output_file: Output file path for the HTML visualization
    """
    # Get feature names
    feature_names = feature_extract.get_feature_names()
    
    # Validate indices
    if len(feature_indices) != 3:
        logger.error("Must provide exactly 3 feature indices for 3D plotting")
        return
    
    # Collect all songs with features
    songs_with_features = []
    playlist_map = {}  # To track which playlists each song belongs to
    
    for playlist in library.playlists:
        for song in playlist.songs:
            # Check if song has features and all required indices are available
            if (hasattr(song, 'features') and song.features and 
                len(song.features) > max(feature_indices)):
                
                # Track playlist membership
                if song.id not in playlist_map:
                    playlist_map[song.id] = []
                    songs_with_features.append(song)
                playlist_map[song.id].append(playlist.name)
    
    if not songs_with_features:
        logger.warning("No songs with features found to plot")
        return
    
    logger.info(f"Plotting {len(songs_with_features)} songs with features")
    
    # Prepare data for plotting
    titles = [f"{song.title} by {song.artist}" for song in songs_with_features]
    features_array = np.array([[song.features[i] for i in feature_indices] for song in songs_with_features])
    
    # Get playlist labels for coloring
    playlist_labels = [', '.join(playlist_map.get(song.id, ['Unknown'])) for song in songs_with_features]
    
    # Get axis labels
    axis_labels = [feature_names[i] if i < len(feature_names) else f"Feature {i}" for i in feature_indices]
    
    # Prepare a DataFrame for better plotting
    import pandas as pd
    plot_df = pd.DataFrame({
        'x': features_array[:, 0],
        'y': features_array[:, 1],
        'z': features_array[:, 2],
        'title': titles,
        'playlist': playlist_labels
    })
    
    # Create 3D scatter plot with enhanced styling
    fig = px.scatter_3d(
        plot_df,
        x='x',
        y='y',
        z='z',
        color='playlist',
        hover_name='title',  # Show title on hover
        labels={
            'x': axis_labels[0],
            'y': axis_labels[1], 
            'z': axis_labels[2],
            'playlist': 'Playlist'  # Better legend title
        },
        title='Music Feature Visualization',
        color_discrete_sequence=px.colors.qualitative.Bold,  # More vibrant color scheme
    )
    
    # Improve plot aesthetics
    fig.update_traces(
        marker=dict(
            size=6,                 # Slightly larger markers
            opacity=0.8,            # Slight transparency
            line=dict(width=1, color='DarkSlateGrey'),  # Marker borders
            symbol='circle'
        ),
        selector=dict(mode='markers')
    )
    
    # Enhanced layout
    fig.update_layout(
        scene=dict(
            xaxis=dict(backgroundcolor="rgb(230, 230, 250)", gridcolor="white", showbackground=True),
            yaxis=dict(backgroundcolor="rgb(230, 230, 250)", gridcolor="white", showbackground=True),
            zaxis=dict(backgroundcolor="rgb(230, 230, 250)", gridcolor="white", showbackground=True),
        ),
        legend=dict(
            title_font=dict(size=14),
            font=dict(size=12),
            itemsizing='constant'
        ),
        margin=dict(r=20, l=10, b=10, t=50),
        title_font=dict(size=20),
    )
    
    # Save the plot to an HTML file
    fig.write_html(output_file)
    logger.info(f"Plot saved to: {output_file}")
    
    # Show the plot
    fig.show()

def plot_features_matrix(library, output_file=VISUALIZATION_MATRIX_PATH):
    """Create a matrix of scatter plots for all combinations of features"""
    # Get feature names
    feature_names = feature_extract.get_feature_names()
    
    # Collect all songs with features
    songs_with_features = []
    playlist_map = {}  # To track which playlists each song belongs to
    
    for playlist in library.playlists:
        for song in playlist.songs:
            # Only use songs with the base features (first 6)
            if hasattr(song, 'features') and song.features and len(song.features) >= 6:
                # Track playlist membership
                if song.id not in playlist_map:
                    playlist_map[song.id] = []
                    songs_with_features.append(song)
                playlist_map[song.id].append(playlist.name)
    
    if not songs_with_features:
        logger.warning("No songs with features found to plot")
        return
    
    logger.info(f"Plotting feature matrix for {len(songs_with_features)} songs")
    
    # Prepare data for plotting
    titles = [f"{song.title} by {song.artist}" for song in songs_with_features]
    
    # Get playlist labels for coloring
    playlist_labels = [', '.join(playlist_map.get(song.id, ['Unknown'])) for song in songs_with_features]
    
    # Create a dataframe for plotting
    import pandas as pd
    df = pd.DataFrame()
    
    # Add song info
    df['Title'] = titles
    df['Playlist'] = playlist_labels
    
    # Add features (only use the first 6 to avoid too many plots)
    for i in range(min(6, len(feature_names))):
        feature_data = [song.features[i] for song in songs_with_features]
        df[feature_names[i]] = feature_data
    
    # Create scatter plot matrix
    fig = px.scatter_matrix(
        df,
        dimensions=feature_names[:6],
        color='Playlist',
        hover_data=['Title'],
        title='Music Feature Relationships'
    )
    
    # Improve formatting
    fig.update_traces(diagonal_visible=False)
    
    # Save the plot to an HTML file
    fig.write_html(output_file)
    logger.info(f"Feature matrix plot saved to: {output_file}")
    
    # Show the plot
    fig.show()

def generate_library_report(library):
    """Generate and display a report of library statistics"""
    stats = library.generate_library_stats()
    
    # Display basic stats
    logger.info("=== LIBRARY STATISTICS ===")
    logger.info(f"Total songs: {stats['total_songs']}")
    logger.info(f"Total playlists: {stats['total_playlists']}")
    logger.info(f"Songs with features: {stats['songs_with_features']} ({stats['songs_with_features']/stats['total_songs']*100:.1f}%)")
    logger.info(f"Songs with MP3 files: {stats['songs_with_mp3']} ({stats['songs_with_mp3']/stats['total_songs']*100:.1f}%)")
    
    if stats['total_duration_hours'] > 0:
        logger.info(f"Total duration: {stats['total_duration_hours']:.1f} hours")
    
    # Display playlist stats
    logger.info("\n=== PLAYLIST STATISTICS ===")
    for name, playlist_stats in stats['playlists'].items():
        logger.info(f"\nPlaylist: {name}")
        logger.info(f"  Songs: {playlist_stats['songs_count']}")
        percentage_with_features = playlist_stats['songs_with_features']/playlist_stats['songs_count']*100 if playlist_stats['songs_count'] > 0 else 0
        percentage_with_mp3 = playlist_stats['songs_with_mp3']/playlist_stats['songs_count']*100 if playlist_stats['songs_count'] > 0 else 0
        logger.info(f"  Songs with features: {playlist_stats['songs_with_features']} ({percentage_with_features:.1f}%)")
        logger.info(f"  Songs with MP3 files: {playlist_stats['songs_with_mp3']} ({percentage_with_mp3:.1f}%)")
    
    # Create a stats file
    stats_file = os.path.join(musicnet_library.LIBRARY_PATH, "library_stats.json")
    with open(stats_file, 'w') as f:
        import json
        json.dump(stats, f, indent=2)
    
    logger.info(f"\nStatistics saved to {stats_file}")

def main():
    """Main function to run the music library and plotting process"""
    logger.info("Starting MusicNet - Music Analysis and Visualization Platform")
    
    # Load or create library with playlists from settings
    library = ensure_library()
    
    # Generate library stats if requested
    if GENERATE_STATISTICS:
        generate_library_report(library)
    
    # Step 1: Download songs if not skipped
    if not SKIP_DOWNLOAD:
        library = download_songs(library)
    
    # Step 2: Extract features if not skipped
    if not SKIP_EXTRACTION:
        library = extract_features(
            library, 
            parallel=USE_PARALLEL_PROCESSING,
            max_processes=NUM_PROCESSES,
            force=FORCE_EXTRACTION
        )
        # Log a message explaining how to speed up future runs
        if not FORCE_EXTRACTION:
            logger.info("NOTE: Features are cached between runs. Set FORCE_EXTRACTION=True to recalculate.")
    
    # Step 3: Create visualizations unless skipped
    if not SKIP_PLOTS:
        # 3D visualization
        plot_features_3d(library, feature_indices=FEATURE_INDICES)
        
        # Feature matrix if requested
        if GENERATE_MATRIX_PLOT:
            plot_features_matrix(library)
    
    # Close database connection
    library.close()
    
    logger.info("MusicNet execution completed successfully")

if __name__ == "__main__":
    main()
