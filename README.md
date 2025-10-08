# YouTube Audio Music Analysis

A Python proof-of-concept tool for downloading YouTube audio and analyzing music characteristics using librosa.

## Features

- **YouTube Audio Download**: Downloads audio from YouTube videos using yt-dlp
- **Music Analysis**: Analyzes audio files for various music characteristics including:
  - Tempo (BPM)
  - Spectral features
  - Harmonic and percussive components
  - MFCC features
  - Chroma features for key detection
  - Music classification heuristics

## Requirements

- Python 3.8+
- FFmpeg (for audio processing)

## Installation

1. Clone this repository:
```bash
git clone <your-repo-url>
cd youtube-audio-analysis
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Install FFmpeg (if not already installed):
```bash
# On macOS with Homebrew
brew install ffmpeg

# On Ubuntu/Debian
sudo apt update
sudo apt install ffmpeg

# On Windows
# Download from https://ffmpeg.org/download.html
```

## Usage

### Basic Usage

```python
from youtube_audio_poc import youtube_audio_music_analysis

# Analyze a YouTube video
result = youtube_audio_music_analysis("https://www.youtube.com/watch?v=your-video-id")

if result['success']:
    print(f"Audio saved to: {result['audio_path']}")
    print(f"Music probability: {result['analysis']['music_probability']:.2%}")
else:
    print(f"Error: {result['error']}")
```

### Command Line Usage

```bash
python youtube_audio_poc.py
```

## Project Structure

```
├── youtube_audio_poc.py    # Main analysis script
├── requirements.txt        # Python dependencies
├── poc/                   # Downloaded audio files (ignored by git)
└── README.md              # This file
```

## Features Analysis

The tool analyzes various audio characteristics:

- **Tempo**: Detects BPM using beat tracking
- **Spectral Features**: Analyzes frequency content
- **Harmonic/Percussive Separation**: Distinguishes melodic vs rhythmic content
- **MFCC Features**: Mel-frequency cepstral coefficients for timbre analysis
- **Chroma Features**: Pitch class analysis for key detection
- **Music Classification**: Heuristic-based music probability scoring

## Troubleshooting

### Robot Detection
If you encounter robot detection errors:
- Try using a VPN
- Wait a few minutes between requests
- Use different YouTube URLs
- Open YouTube in your browser first

### FFmpeg Issues
Make sure FFmpeg is properly installed and accessible from your PATH.

## License

This project is for educational and research purposes.
