# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

Proof-of-concept Python tools for generating bass guitar ASCII tablature from audio or MIDI. No LLMs, no agents — pure signal-processing pipelines.

## Setup

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
pip install pretty_midi          # missing from requirements.txt but needed by midi_to_tab.py
brew install ffmpeg              # macOS; required by yt_dlp for WAV conversion
```

## Running the Scripts

```bash
# WAV file → ASCII tab (Demucs + librosa pipeline)
python bass_tab_generator.py <input.wav>

# MIDI file → ASCII tab (pretty_midi pipeline, better timing accuracy)
python midi_to_tab.py <input.mid> [--tracks 0,1] [--output out.txt] [--measures-per-row 4] [--subdivisions 8]

# YouTube URL → downloaded WAV + music feature analysis (edit hardcoded URL in __main__ first)
python youtube_audio_poc.py
```

There are no tests, no linter config, and no build step.

## Architecture

### Two separate tab-generation pipelines (not integrated)

**`bass_tab_generator.py` — `BassTabGenerator` class**
WAV → bass stem → pitches → technique labels → ASCII tab
1. Loads WAV via librosa (resampled to 22050 Hz)
2. Runs `demucs --two-stems bass` as a subprocess; falls back to librosa HPSS if Demucs fails
3. Detects pitches with `librosa.piptrack` (E1–C4 range)
4. Classifies technique by spectral centroid: >3000 Hz = Pop, >2000 Hz = Slap, else Normal
5. Maps MIDI notes to fret positions, deduplicates consecutive identical notes, renders ASCII tab

**`midi_to_tab.py` — `BassTabRenderer` class**
MIDI → measure grid → ASCII tab
1. Parses MIDI with `pretty_midi`; auto-detects bass tracks by program 32–39
2. Snaps notes to a beat grid (default: 8 subdivisions/measure = eighth notes in 4/4)
3. Renders 4 measures per row with measure numbers

**`youtube_audio_poc.py`** — standalone utility; downloads audio via `yt_dlp`, runs librosa feature extraction, prints stats. Has no tab output and no CLI argument parsing.

### Bass tuning (shared convention)
```
G=43, D=38, A=33, E=28  (MIDI note numbers for open strings)
```

The two scripts use **different** note-to-fret strategies:
- `bass_tab_generator.py`: greedy G→E, first string with fret ≤ 24 (can produce unplayable positions)
- `midi_to_tab.py`: scores candidates, prefers frets 0–9 then lower strings (more playable)

### Demucs output layout
Demucs writes to `./separated/<model>/<song_stem>/bass.wav` relative to CWD. Always run from the project root. The script searches for the model directory in `['htdemucs', 'htdemucs_ft', 'mdx_extra_q', 'mdx']`; the default model in use is `htdemucs`.

## Known Issues / Gotchas

- **`pretty_midi` not in `requirements.txt`** — `midi_to_tab.py` will fail without it.
- **`--save-bass` flag has no effect** — `main()` in `bass_tab_generator.py` unconditionally passes a save path, so bass audio is always written to `<stem>_bass.wav`.
- **Tab capped at 40 notes** — `generate_tab()` silently truncates `filtered_notes[:40]`.
- **`is_bright` always `False`** in `youtube_audio_poc.py` — the check compares `mean > mean * 1.2`, which is never true.
- **No timing in `bass_tab_generator.py` output** — notes are sequential only, no rhythm or measures.
