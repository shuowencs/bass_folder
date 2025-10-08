import os
import tempfile
import librosa
import numpy as np
import yt_dlp
from pathlib import Path
import logging

# Set up logging for detailed output
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def download_youtube_audio(url, output_dir=None):
    """
    Download audio from YouTube URL using yt-dlp
    
    Args:
        url (str): YouTube URL
        output_dir (str): Directory to save the audio file (optional)
    
    Returns:
        str: Path to the downloaded audio file
    """
    print(f"🎵 Starting YouTube audio download from: {url}")
    
    if output_dir is None:
        # Default to a persistent project subfolder: <project_root>/poc
        project_root = Path(__file__).resolve().parent
        output_dir = os.path.join(project_root, "poc")
        os.makedirs(output_dir, exist_ok=True)
        print(f"📁 Output directory set to: {output_dir}")
    
    # Configure yt-dlp options with anti-bot measures and FFmpeg post-processing
    ydl_opts = {
        'format': 'bestaudio/best',
        # Always produce wav using FFmpegAudioConvertor
        'outtmpl': os.path.join(output_dir, '%(title)s.%(ext)s'),
        'postprocessors': [
            {
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'wav',
                'preferredquality': '0',
            }
        ],
        'noplaylist': True,
        'quiet': False,
        'no_warnings': False,
        # Anti-bot measures
        'cookiesfrombrowser': ('chrome',),  # Use browser cookies
        'sleep_interval': 1,  # Add delay between requests
        'max_sleep_interval': 5,
        'sleep_interval_subtitles': 1,
        # User agent to appear more like a real browser
        'http_headers': {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        },
        # Retry options
        'retries': 3,
        'fragment_retries': 3,
        'skip_unavailable_fragments': True,
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            print("📥 Extracting video information...")
            info = ydl.extract_info(url, download=False)
            title = info.get('title', 'Unknown Title')
            duration = info.get('duration', 0)
            
            print(f"📋 Video Title: {title}")
            print(f"⏱️  Duration: {duration} seconds")
            
            print("⬇️  Starting audio download...")
            ydl.download([url])
            
            # Find the downloaded file (check multiple extensions, prefer wav)
            exts = ('.wav', '.m4a', '.mp3', '.webm', '.opus')
            candidates = [f for f in os.listdir(output_dir) if f.lower().endswith(exts)]
            # Prefer .wav if present, else most recent candidate
            wavs = [f for f in candidates if f.lower().endswith('.wav')]
            if wavs:
                target_file = wavs[0]
            elif candidates:
                # pick newest by mtime
                candidates.sort(key=lambda f: os.path.getmtime(os.path.join(output_dir, f)), reverse=True)
                target_file = candidates[0]
            else:
                target_file = None

            if target_file:
                audio_path = os.path.join(output_dir, target_file)
                print(f"✅ Audio downloaded successfully: {audio_path}")
                return audio_path
            else:
                raise FileNotFoundError("No audio file found after download")
                
    except Exception as e:
        error_msg = str(e)
        print(f"❌ Error downloading audio: {error_msg}")
        
        # Check for specific robot detection errors
        if "robot" in error_msg.lower() or "captcha" in error_msg.lower() or "verify" in error_msg.lower():
            print("\n🤖 ROBOT DETECTION DETECTED!")
            print("YouTube is asking you to verify you're not a robot.")
            print("\n💡 SOLUTIONS:")
            print("1. Try downloading from a different IP/VPN")
            print("2. Use a different YouTube URL")
            print("3. Wait a few minutes and try again")
            print("4. Try opening YouTube in your browser first")
            print("5. Consider using a different video that's less restricted")
            
        elif "private" in error_msg.lower() or "unavailable" in error_msg.lower():
            print("\n🔒 VIDEO ACCESS ISSUE!")
            print("The video might be private, geo-restricted, or unavailable.")
            print("Try a different public YouTube video.")
            
        elif "age" in error_msg.lower() or "restricted" in error_msg.lower():
            print("\n👶 AGE-RESTRICTED VIDEO!")
            print("This video has age restrictions. Try a different video.")
            
        else:
            print(f"\n🔧 GENERAL ERROR: {error_msg}")
            print("Try checking your internet connection or using a different URL.")
            # If ffmpeg is missing, hint how to install
            if 'ffmpeg' in error_msg.lower() or 'postprocessor' in error_msg.lower():
                print("\n🛠️ It looks like FFmpeg might be missing.")
                print("Install on macOS with Homebrew:")
                print("   brew install ffmpeg")
                print("Then re-run: pip install -r requirements.txt and try again.")
            
        raise

def analyze_music_characteristics(audio_path):
    """
    Analyze audio file for music characteristics using librosa
    
    Args:
        audio_path (str): Path to the audio file
    
    Returns:
        dict: Analysis results including tempo, key, genre indicators, etc.
    """
    print(f"🔍 Analyzing music characteristics for: {audio_path}")
    
    try:
        # Load audio file
        print("📊 Loading audio file...")
        y, sr = librosa.load(audio_path, sr=None)
        duration = len(y) / sr
        print(f"📈 Audio loaded: {len(y)} samples, {sr} Hz sample rate, {duration:.2f}s duration")
        
        # Basic audio features
        print("🎼 Extracting basic audio features...")
        
        # Spectral features
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        zero_crossing_rate = librosa.feature.zero_crossing_rate(y)[0]
        
        # Rhythm features
        print("🥁 Analyzing rhythm and tempo...")
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
        
        # Harmonic and percussive components
        print("🎵 Separating harmonic and percussive components...")
        y_harmonic, y_percussive = librosa.effects.hpss(y)
        
        # MFCC features (commonly used for music analysis)
        print("🎤 Extracting MFCC features...")
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        
        # Chroma features (for key detection)
        print("🎹 Analyzing chroma features for key detection...")
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        
        # Tonnetz features (harmonic network)
        print("🌐 Computing Tonnetz features...")
        tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr)
        
        # Music genre indicators
        print("🎭 Computing genre indicators...")
        
        # Calculate various statistics
        analysis_results = {
            'file_path': audio_path,
            'duration_seconds': duration,
            'sample_rate': sr,
            'tempo_bpm': float(tempo),
            'spectral_centroid_mean': float(np.mean(spectral_centroids)),
            'spectral_centroid_std': float(np.std(spectral_centroids)),
            'spectral_rolloff_mean': float(np.mean(spectral_rolloff)),
            'zero_crossing_rate_mean': float(np.mean(zero_crossing_rate)),
            'zero_crossing_rate_std': float(np.std(zero_crossing_rate)),
            'mfcc_mean': [float(np.mean(mfcc)) for mfcc in mfccs],
            'mfcc_std': [float(np.std(mfcc)) for mfcc in mfccs],
            'chroma_mean': [float(np.mean(c)) for c in chroma],
            'harmonic_energy': float(np.mean(y_harmonic**2)),
            'percussive_energy': float(np.mean(y_percussive**2)),
            'energy_ratio': float(np.mean(y_harmonic**2) / np.mean(y_percussive**2)) if np.mean(y_percussive**2) > 0 else 0,
        }
        
        # Music classification heuristics
        print("🎯 Computing music classification heuristics...")
        
        # High tempo suggests dance/electronic music
        is_high_tempo = tempo > 120
        
        # High spectral centroid suggests bright/high-frequency content
        is_bright = np.mean(spectral_centroids) > np.mean(spectral_centroids) * 1.2
        
        # High harmonic energy suggests melodic content
        is_melodic = analysis_results['harmonic_energy'] > analysis_results['percussive_energy']
        
        # Low zero-crossing rate suggests tonal content
        is_tonal = np.mean(zero_crossing_rate) < 0.1
        
        analysis_results.update({
            'is_high_tempo': is_high_tempo,
            'is_bright': is_bright,
            'is_melodic': is_melodic,
            'is_tonal': is_tonal,
            'music_probability': float(
                (is_melodic * 0.4) + 
                (is_tonal * 0.3) + 
                (is_bright * 0.2) + 
                (is_high_tempo * 0.1)
            )
        })
        
        print("✅ Music analysis completed successfully!")
        return analysis_results
        
    except Exception as e:
        print(f"❌ Error analyzing audio: {str(e)}")
        raise

def youtube_audio_music_analysis(url, output_dir=None):
    """
    Main proof-of-concept function that downloads YouTube audio and analyzes it for music characteristics
    
    Args:
        url (str): YouTube URL
        output_dir (str): Directory to save the audio file (optional)
    
    Returns:
        dict: Contains audio file path and analysis results
    """
    print("=" * 60)
    print("🎵 YOUTUBE AUDIO MUSIC ANALYSIS PROOF OF CONCEPT")
    print("=" * 60)
    
    try:
        # Step 1: Download audio
        print("\n📥 STEP 1: DOWNLOADING AUDIO")
        print("-" * 30)
        audio_path = download_youtube_audio(url, output_dir)
        
        # Step 2: Analyze music characteristics
        print("\n🔍 STEP 2: ANALYZING MUSIC CHARACTERISTICS")
        print("-" * 40)
        analysis_results = analyze_music_characteristics(audio_path)
        
        # Step 3: Display results
        print("\n📊 STEP 3: ANALYSIS RESULTS")
        print("-" * 25)
        print(f"📁 Audio File: {analysis_results['file_path']}")
        print(f"⏱️  Duration: {analysis_results['duration_seconds']:.2f} seconds")
        print(f"🎵 Tempo: {analysis_results['tempo_bpm']:.1f} BPM")
        print(f"🎼 Spectral Centroid: {analysis_results['spectral_centroid_mean']:.2f}")
        print(f"🎹 Zero Crossing Rate: {analysis_results['zero_crossing_rate_mean']:.4f}")
        print(f"🎭 Harmonic Energy: {analysis_results['harmonic_energy']:.6f}")
        print(f"🥁 Percussive Energy: {analysis_results['percussive_energy']:.6f}")
        print(f"📈 Energy Ratio (H/P): {analysis_results['energy_ratio']:.2f}")
        
        print("\n🎯 MUSIC CLASSIFICATION:")
        print(f"   High Tempo (>120 BPM): {'✅' if analysis_results['is_high_tempo'] else '❌'}")
        print(f"   Bright Sound: {'✅' if analysis_results['is_bright'] else '❌'}")
        print(f"   Melodic Content: {'✅' if analysis_results['is_melodic'] else '❌'}")
        print(f"   Tonal Content: {'✅' if analysis_results['is_tonal'] else '❌'}")
        print(f"🎵 Music Probability: {analysis_results['music_probability']:.2%}")
        
        print("\n✅ ANALYSIS COMPLETE!")
        print("=" * 60)
        
        return {
            'audio_path': audio_path,
            'analysis': analysis_results,
            'success': True
        }
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        print("=" * 60)
        return {
            'audio_path': None,
            'analysis': None,
            'success': False,
            'error': str(e)
        }

def get_suggested_test_urls():
    """
    Returns a list of suggested YouTube URLs for testing that are less likely to trigger robot detection
    """
    return [
        'https://www.youtube.com/watch?v=v_O7LDxrPvA&list=RDv_O7LDxrPvA&start_radio=1',
       # "https://www.youtube.com/watch?v=dQw4w9WgXcQ",  # Rick Roll (classic)
       # "https://www.youtube.com/watch?v=9bZkp7q19f0",  # PSY - GANGNAM STYLE
       # "https://www.youtube.com/watch?v=kJQP7kiw5Fk",  # Luis Fonsi - Despacito
      #  "https://www.youtube.com/watch?v=fJ9rUzIMcZQ",  # Queen - Bohemian Rhapsody
      #  "https://www.youtube.com/watch?v=L_jWHffIx5E",  # Smells Like Teen Spirit
    ]

# Example usage
if __name__ == "__main__":
    print("🚀 Starting YouTube Audio Music Analysis Proof of Concept")
    
    # Get suggested URLs
    suggested_urls = get_suggested_test_urls()
    print("\n📋 Suggested test URLs (less likely to trigger robot detection):")
    for i, url in enumerate(suggested_urls, 1):
        print(f"   {i}. {url}")
    
    # Use the first suggested URL
    example_url = suggested_urls[0]
    print(f"\n🔗 Testing with URL: {example_url}")
    
    result = youtube_audio_music_analysis(example_url)
    
    if result['success']:
        print(f"\n🎉 Success! Audio saved to: {result['audio_path']}")
        print(f"📊 Analysis completed with {result['analysis']['music_probability']:.2%} music probability")
    else:
        print(f"\n💥 Failed: {result['error']}")
        
        # If robot detection, suggest trying other URLs
        if "robot" in str(result.get('error', '')).lower():
            print("\n🔄 ROBOT DETECTION - Try these alternative URLs:")
            for i, url in enumerate(suggested_urls[1:], 2):
                print(f"   {i}. {url}")
            print("\n💡 You can also try:")
            print("   - Using a VPN")
            print("   - Waiting a few minutes")
            print("   - Opening YouTube in your browser first")