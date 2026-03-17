# Bass Tab Generator POC

A simple proof-of-concept that generates bass tablature with slap/pop notation from audio files.

## Features

- Bass frequency extraction using harmonic-percussive separation
- Pitch detection (MIDI note conversion)
- Slap/Pop technique detection based on spectral analysis
- ASCII tablature generation
- Simple, sequential pipeline (no agents)

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
python bass_tab_generator.py <input.wav>
```

**Example:**
```bash
python bass_tab_generator.py my_song.wav
```

This will:
1. Load the audio file
2. Extract bass frequencies
3. Detect pitches and convert to MIDI notes
4. Detect slap/pop techniques
5. Generate ASCII tab
6. Save output to `my_song_tab.txt`

## Output Format

```
Bass Tab
============================================================
Legend: S = Slap, P = Pop, - = Normal fingerstyle
============================================================
G|----------5-7------|
D|5-7-8-------------|
A|------------------|
E|------------------|
T|- - S P - - S ----|
============================================================
```

## How It Works

### 1. Bass Extraction
Uses librosa's harmonic-percussive separation to isolate melodic content, then focuses on low frequencies typical of bass guitar (E1-C4).

### 2. Pitch Detection
Uses librosa's `piptrack` to detect fundamental frequencies and convert to MIDI notes.

### 3. Technique Detection
Analyzes spectral characteristics:
- **Pop (P)**: Very bright spectrum (>3000 Hz centroid)
- **Slap (S)**: Bright spectrum (>2000 Hz centroid)
- **Normal (-)**: Balanced spectrum

### 4. Tab Generation
Maps MIDI notes to fret positions on standard 4-string bass (E-A-D-G tuning).

## Limitations (POC)

⚠️ This is a simple POC with known limitations:
- Bass extraction is basic (no Demucs/Spleeter)
- Timing/rhythm is not captured (notes shown in sequence only)
- Slap/pop detection is heuristic-based (70-80% accuracy)
- String choice uses simple "lowest fret" algorithm
- No support for articulations (slides, hammer-ons, etc.)

## Next Steps

To improve this POC:
1. Integrate Demucs for better bass separation
2. Add rhythm/timing detection
3. Improve slap/pop classifier with ML model
4. Optimize string choice algorithm
5. Add more articulations (H, P, /, \, x)

## Requirements

- Python 3.8+
- librosa
- numpy
- soundfile
