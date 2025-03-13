import musicnet_library
import librosa
import numpy as np
import os
import warnings
import multiprocessing
from multiprocessing import Pool, Queue, Manager
import time
import logging
from functools import partial
import traceback

# Configuration for multiprocessing
MAX_PROCESSES = 10  # Maximum number of parallel extraction processes
CHUNK_SIZE = 5     # Number of songs per process chunk
TIMEOUT = 300      # Timeout for feature extraction (seconds)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join('library', 'feature_extraction.log'))
    ]
)
logger = logging.getLogger('feature_extract')


def extract_features(song, duration=60, sr=22050, detailed=True) -> list:
    """
    Extract audio features from a song.
    
    Args:
        song: Song object containing path to MP3 file
        duration: Duration in seconds to analyze (default: 60s)
        sr: Sample rate for analysis (default: 22050Hz)
        detailed: Whether to extract extended features (default: True)
        
    Returns:
        List of extracted features
    """
    logger.info(f"Extracting features for {song.title} by {song.artist}")
    
    # Check if the song has been downloaded
    if song.path_of_mp3 is None or not os.path.exists(song.path_of_mp3):
        logger.warning(f"Song {song.title} hasn't been downloaded yet.")
        return []
    
    try:
        # Load the audio file with specified duration (first minute by default)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            y, sr = librosa.load(song.path_of_mp3, sr=sr, duration=duration)
        
        # Early termination if the audio is silent or corrupted
        if np.all(y == 0) or np.mean(np.abs(y)) < 1e-4:
            logger.warning(f"Audio for {song.title} appears to be silent or corrupted.")
            return []
        
        # Base features (always extracted)
        # ==============================
        
        # Extract BPM (Tempo)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        bpm = float(tempo)
        
        # Extract spectral centroid (better frequency indicator than simple dominant frequency)
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        
        # Extract dominant frequency with proper bin analysis
        S = np.abs(librosa.stft(y))
        freqs = librosa.fft_frequencies(sr=sr)
        # Get the frequency with the highest magnitude across time
        freq_magnitudes = np.mean(S, axis=1)
        if np.sum(freq_magnitudes) > 0:  # Prevent division by zero
            # Normalize and find weighted average
            freq_magnitudes = freq_magnitudes / np.sum(freq_magnitudes)
            dominant_freq = np.sum(freqs * freq_magnitudes)
        else:
            dominant_freq = 0.0
        
        # Extract spectral bandwidth (width of the frequency band containing energy)
        spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
        
        # Extract pulse clarity
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        pulse_clarity = float(np.mean(onset_env))
        
        # Extract noise (using spectral contrast)
        spectral_contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr))
        noise = float(spectral_contrast)
        
        # Create base feature list
        features = [bpm, dominant_freq, spectral_centroid, spectral_bandwidth, pulse_clarity, noise]
        
        # Extended features (when detailed=True)
        # ====================================
        if detailed:
            # Zero Crossing Rate (percussiveness indicator)
            zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(y))
            
            # RMS Energy (volume/loudness indicator)
            rms_energy = np.mean(librosa.feature.rms(y=y))
            
            # Spectral Rolloff (brightness indicator)
            spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
            
            # Spectral Flatness (tonal vs. noisy indicator)
            spectral_flatness = np.mean(librosa.feature.spectral_flatness(y=y))
            
            # MFCCs (timbre indicators) - take the means of the first 13 coefficients
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            mfcc_means = np.mean(mfccs, axis=1)
            
            # Chroma Features (harmonic content indicators)
            chroma = librosa.feature.chroma_stft(y=y, sr=sr)
            chroma_means = np.mean(chroma, axis=1)
            
            # Harmonic-Percussive Source Separation
            y_harmonic, y_percussive = librosa.effects.hpss(y)
            harmonic_energy = np.mean(y_harmonic**2)
            percussive_energy = np.mean(y_percussive**2)
            harmonic_percussive_ratio = harmonic_energy / (percussive_energy + 1e-8)  # Prevent division by zero
            
            # Add extended features to the list
            features.extend([
                zero_crossing_rate, 
                rms_energy, 
                spectral_rolloff, 
                spectral_flatness, 
                harmonic_percussive_ratio
            ])
            
            # Add MFCCs
            features.extend(mfcc_means.tolist())
            
            # Add Chroma features
            features.extend(chroma_means.tolist())
        
        # Log basic feature extraction results
        logger.info(f"Features extracted: BPM={bpm:.1f}, Dom.Freq={dominant_freq:.1f}Hz, "
                   f"Centroid={spectral_centroid:.1f}Hz, Bandwidth={spectral_bandwidth:.1f}Hz, "
                   f"Pulse={pulse_clarity:.3f}, Noise={noise:.3f}")
        
        return features
        
    except Exception as e:
        logger.error(f"Error extracting features for {song.title}: {e}")
        logger.debug(traceback.format_exc())
        return []


def _extract_worker(songs_chunk, results_dict, lock, counter, total_songs):
    """Worker function for parallel feature extraction"""
    extracted_features = {}
    
    for song in songs_chunk:
        try:
            # Extract features for this song
            features = extract_features(song)
            
            # Store the results
            if features:
                extracted_features[song.id] = features
            
            # Update counter with lock to avoid race conditions
            with lock:
                counter.value += 1
                progress = counter.value / total_songs * 100
                logger.info(f"Progress: {counter.value}/{total_songs} songs processed ({progress:.1f}%)")
        
        except Exception as e:
            logger.error(f"Error in extraction worker for {song.title}: {e}")
            logger.debug(traceback.format_exc())
    
    # Return all extracted features from this chunk
    return extracted_features


def _divide_chunks(songs, chunk_size):
    """Divide songs into chunks for parallel processing"""
    for i in range(0, len(songs), chunk_size):
        yield songs[i:i + chunk_size]


def parallel_extract_features(library, chunk_size=CHUNK_SIZE, max_processes=MAX_PROCESSES):
    """
    Extract features for all songs in the library using parallel processing.
    
    Args:
        library: The music library containing songs
        chunk_size: Number of songs to process in each worker
        max_processes: Maximum number of parallel processes to use
    
    Returns:
        Dictionary mapping song IDs to their extracted features
    """
    start_time = time.time()
    logger.info(f"Starting parallel feature extraction with {max_processes} processes")
    
    # Get all songs that need feature extraction
    songs_to_process = []
    for playlist in library.playlists:
        for song in playlist.songs:
            if len(song.features) == 0 and song.path_of_mp3 and os.path.exists(song.path_of_mp3):
                songs_to_process.append(song)
    
    total_songs = len(songs_to_process)
    if total_songs == 0:
        logger.info("No songs require feature extraction")
        return {}
    
    logger.info(f"Found {total_songs} songs requiring feature extraction")
    
    # Create chunks of songs
    song_chunks = list(_divide_chunks(songs_to_process, chunk_size))
    
    # Use multiprocessing manager for shared objects
    with Manager() as manager:
        # Create shared objects
        results_dict = manager.dict()  # Shared dict for results
        lock = manager.Lock()  # Lock for counter updates
        counter = manager.Value('i', 0)  # Shared counter for progress tracking
        
        # Prepare worker function with partial application
        worker_func = partial(_extract_worker, 
                             results_dict=results_dict, 
                             lock=lock, 
                             counter=counter, 
                             total_songs=total_songs)
        
        # Create process pool and map chunks to workers
        with Pool(processes=min(max_processes, len(song_chunks))) as pool:
            # Execute workers and collect results
            chunk_results = pool.map(worker_func, song_chunks)
        
        # Combine results from all chunks
        all_results = {}
        for result_dict in chunk_results:
            all_results.update(result_dict)
    
    # Calculate statistics
    success_count = len(all_results)
    elapsed_time = time.time() - start_time
    songs_per_second = total_songs / elapsed_time if elapsed_time > 0 else 0
    
    logger.info(f"Feature extraction complete: {success_count}/{total_songs} songs processed successfully")
    logger.info(f"Extraction took {elapsed_time:.1f} seconds ({songs_per_second:.2f} songs/second)")
    
    return all_results


def update_library_with_features(library, feature_results):
    """Update the library with extracted features"""
    update_count = 0
    
    for playlist in library.playlists:
        for song in playlist.songs:
            if song.id in feature_results:
                song.features = feature_results[song.id]
                update_count += 1
    
    logger.info(f"Updated {update_count} songs with new features")
    return library


def get_feature_names(detailed=True):
    """Return the names of features in the order they are extracted"""
    # Base feature names
    feature_names = [
        "BPM", 
        "Dominant Frequency", 
        "Spectral Centroid", 
        "Spectral Bandwidth", 
        "Pulse Clarity", 
        "Noise"
    ]
    
    # Extended feature names
    if detailed:
        feature_names.extend([
            "Zero Crossing Rate",
            "RMS Energy",
            "Spectral Rolloff",
            "Spectral Flatness",
            "Harmonic/Percussive Ratio"
        ])
        
        # MFCC names
        for i in range(13):
            feature_names.append(f"MFCC_{i+1}")
            
        # Chroma names
        for i in range(12):
            feature_names.append(f"Chroma_{i+1}")
    
    return feature_names
