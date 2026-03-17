"""
Bass Tab Generator POC
======================
Takes a WAV file, extracts bass, detects pitches and generates tab with slap/pop notation.

Usage: python bass_tab_generator.py input.wav
"""

import sys
import numpy as np
import librosa
from pathlib import Path


class BassTabGenerator:
    """Simple bass tab generator with slap/pop detection."""
    
    def __init__(self, sample_rate=22050):
        self.sample_rate = sample_rate
        self.bass_strings = {
            'G': 43,  # G string (MIDI note 43)
            'D': 38,  # D string
            'A': 33,  # A string
            'E': 28   # E string
        }
    
    def load_audio(self, wav_path):
        """Load WAV file."""
        print(f"📂 Loading audio: {wav_path}")
        y, sr = librosa.load(wav_path, sr=self.sample_rate)
        print(f"✅ Loaded {len(y)/sr:.2f} seconds of audio")
        return y, sr
    
    def extract_bass(self, y, sr, save_path=None, use_demucs=True, input_file=None):
        """
        Extract bass from audio.
        
        Parameters:
        -----------
        y : np.array
            Audio time series
        sr : int
            Sample rate
        save_path : str, optional
            If provided, saves the extracted bass to this WAV file
        use_demucs : bool, optional
            If True, use Demucs 4 for high-quality separation (recommended)
            If False, use simple harmonic-percussive separation
        input_file : str, optional
            Original input file path (required if use_demucs=True)
        """
        
        if use_demucs:
            print("🎸 Extracting bass with Demucs 4 (high quality)...")
            
            if input_file is None:
                raise ValueError("input_file required when use_demucs=True")
            
            import subprocess
            import shutil
            from pathlib import Path
            
            # Determine output directory
            # Demucs saves to "separated" in the current working directory
            output_dir = Path("separated")
            
            # Run Demucs command: demucs --two-stems bass input.wav
            try:
                print("   Running: demucs --two-stems bass ...")
                
                cmd = [
                    'demucs',
                    '--two-stems', 'bass',
                    str(input_file)
                ]
                
                # Run the command
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    check=True
                )
                
                # Demucs saves to: separated/<model_name>/<song_name>/bass.wav
                # Default model is 'htdemucs' (without -n flag), but could be 'htdemucs_ft' or others
                song_name = Path(input_file).stem
                
                # Try to find the bass file in common model directories
                possible_models = ['htdemucs', 'htdemucs_ft', 'mdx_extra_q', 'mdx']
                bass_path = None
                
                for model_name in possible_models:
                    potential_path = output_dir / model_name / song_name / 'bass.wav'
                    if potential_path.exists():
                        bass_path = potential_path
                        print(f"   Found bass file in: {model_name}/")
                        break
                
                if bass_path is None:
                    # List what was actually created for debugging
                    print(f"   Searching in: {output_dir}")
                    if output_dir.exists():
                        print(f"   Available model directories:")
                        for item in output_dir.iterdir():
                            if item.is_dir():
                                print(f"      - {item.name}")
                                # Check subdirectories
                                for subitem in item.iterdir():
                                    if subitem.is_dir():
                                        print(f"        - {subitem.name}/")
                    raise FileNotFoundError(
                        f"Demucs output not found. Expected in one of: "
                        f"{[str(output_dir / m / song_name / 'bass.wav') for m in possible_models]}"
                    )
                
                print(f"✅ Demucs extracted bass to: {bass_path}")
                
                # Load the extracted bass audio
                bass_audio, bass_sr = librosa.load(str(bass_path), sr=self.sample_rate)
                
                # Optionally copy to user-specified save path
                if save_path:
                    shutil.copy(bass_path, save_path)
                    print(f"💾 Bass audio copied to: {save_path}")
                
                print("✅ Bass extraction complete")
                return bass_audio
                
            except subprocess.CalledProcessError as e:
                print(f"⚠️  Demucs command failed:")
                print(f"    {e.stderr}")
                print("⚠️  Falling back to simple extraction...")
                use_demucs = False
            except Exception as e:
                print(f"⚠️  Demucs error: {e}")
                print("⚠️  Falling back to simple extraction...")
                use_demucs = False
        
        if not use_demucs:
            print("🎸 Extracting bass frequencies (simple method)...")
            
            # Separate harmonic (melodic) from percussive
            y_harmonic, y_percussive = librosa.effects.hpss(y)
            
            # Focus on bass frequencies
            bass_audio = librosa.effects.preemphasis(y_harmonic, coef=0.97)
            
            # Optionally save to file
            if save_path:
                import soundfile as sf
                sf.write(save_path, bass_audio, sr)
                print(f"💾 Bass audio saved to: {save_path}")
            
            print("✅ Bass extraction complete")
            return bass_audio
    
    def detect_pitches(self, y, sr):
        """Detect pitch over time using librosa's piptrack."""
        print("🎵 Detecting pitches...")
        
        # Pitch detection
        pitches, magnitudes = librosa.piptrack(
            y=y, 
            sr=sr,
            fmin=librosa.note_to_hz('E1'),  # Lowest bass note
            fmax=librosa.note_to_hz('C4')    # Highest typical bass note
        )
        
        # Extract most prominent pitch at each time frame
        pitch_sequence = []
        times = librosa.frames_to_time(np.arange(pitches.shape[1]), sr=sr)
        
        for t_idx in range(pitches.shape[1]):
            # Get the pitch with highest magnitude at this time
            index = magnitudes[:, t_idx].argmax()
            pitch = pitches[index, t_idx]
            
            if pitch > 0:  # Valid pitch detected
                midi_note = librosa.hz_to_midi(pitch)
                pitch_sequence.append({
                    'time': times[t_idx],
                    'midi': midi_note,
                    'magnitude': magnitudes[index, t_idx]
                })
        
        print(f"✅ Detected {len(pitch_sequence)} pitch events")
        return pitch_sequence
    
    def detect_technique(self, y, sr, pitch_sequence):
        """
        Detect slap/pop based on spectral characteristics.
        
        Simple heuristic:
        - Slap: Sharp attack, strong high-frequency content
        - Pop: Very sharp attack, extremely bright spectrum
        - Normal: Smoother attack, balanced spectrum
        """
        print("👋 Detecting slap/pop techniques...")
        
        # Get onset strength (attack sharpness)
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        onset_frames = librosa.onset.onset_detect(
            onset_envelope=onset_env, 
            sr=sr,
            backtrack=True
        )
        onset_times = librosa.frames_to_time(onset_frames, sr=sr)
        
        # Get spectral centroid (brightness)
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        
        techniques = []
        for pitch_info in pitch_sequence:
            time = pitch_info['time']
            
            # Find closest onset
            onset_diffs = np.abs(onset_times - time)
            if len(onset_diffs) > 0:
                closest_onset_idx = onset_diffs.argmin()
                
                # Get spectral brightness at this time
                frame_idx = librosa.time_to_frames(time, sr=sr)
                if frame_idx < len(spectral_centroids):
                    brightness = spectral_centroids[frame_idx]
                    
                    # Simple classification based on brightness
                    if brightness > 3000:  # Very bright
                        technique = 'P'  # Pop
                    elif brightness > 2000:  # Bright
                        technique = 'S'  # Slap
                    else:
                        technique = '-'  # Normal fingerstyle
                    
                    techniques.append(technique)
                else:
                    techniques.append('-')
            else:
                techniques.append('-')
        
        print(f"✅ Technique detection complete")
        return techniques
    
    def midi_to_fret(self, midi_note):
        """Convert MIDI note to (string, fret) position."""
        # Simple algorithm: choose lowest fret position
        for string_name, open_string_midi in sorted(
            self.bass_strings.items(), 
            key=lambda x: x[1], 
            reverse=True
        ):
            if midi_note >= open_string_midi:
                fret = int(round(midi_note - open_string_midi))
                if fret <= 24:  # Reasonable fret range
                    return string_name, fret
        
        return None, None
    
    def generate_tab(self, pitch_sequence, techniques):
        """Generate ASCII bass tablature."""
        print("📝 Generating tablature...")
        
        # Group notes (simplified - just take unique pitches)
        notes = []
        for i, pitch_info in enumerate(pitch_sequence):
            string, fret = self.midi_to_fret(pitch_info['midi'])
            if string and fret is not None:
                technique = techniques[i] if i < len(techniques) else '-'
                notes.append({
                    'string': string,
                    'fret': fret,
                    'technique': technique,
                    'time': pitch_info['time']
                })
        
        # Remove duplicates (keep only significant pitch changes)
        filtered_notes = []
        last_note = None
        for note in notes:
            if (last_note is None or 
                note['string'] != last_note['string'] or 
                note['fret'] != last_note['fret']):
                filtered_notes.append(note)
                last_note = note
        
        # Build tab
        tab_lines = {
            'G': 'G|',
            'D': 'D|',
            'A': 'A|',
            'E': 'E|',
            'T': 'T|'  # Technique line
        }
        
        for note in filtered_notes[:40]:  # Limit to first 40 notes for POC
            fret_str = str(note['fret'])
            padding = '-' * (len(fret_str) + 1)
            
            for string_name in ['G', 'D', 'A', 'E']:
                if string_name == note['string']:
                    tab_lines[string_name] += fret_str + '-'
                else:
                    tab_lines[string_name] += padding
            
            # Add technique notation
            tab_lines['T'] += note['technique'] + (' ' * len(fret_str))
        
        # Close tab lines
        for key in tab_lines:
            tab_lines[key] += '|'
        
        tab = '\n'.join([
            'Bass Tab',
            '=' * 60,
            'Legend: S = Slap, P = Pop, - = Normal fingerstyle',
            '=' * 60,
            tab_lines['G'],
            tab_lines['D'],
            tab_lines['A'],
            tab_lines['E'],
            tab_lines['T'],
            '=' * 60,
        ])
        
        print("✅ Tab generation complete")
        return tab


def main():
    if len(sys.argv) < 2:
        print("Usage: python bass_tab_generator.py <input.wav>")
        print("  --save-bass    Save extracted bass audio as <input>_bass.wav")
        sys.exit(1)
    
    input_file = sys.argv[1]
    save_bass = '--save-bass' in sys.argv
    
    if not Path(input_file).exists():
        print(f"❌ Error: File '{input_file}' not found")
        sys.exit(1)
    
    print("🎸 Bass Tab Generator POC")
    print("=" * 60)
    
    # Initialize generator
    generator = BassTabGenerator()
    
    # Pipeline
    y, sr = generator.load_audio(input_file)
    bass_save_path = Path(input_file).stem + "_bass.wav"

    bass_audio = generator.extract_bass(y, sr, save_path=bass_save_path, use_demucs=True, input_file=input_file)
    pitch_sequence = generator.detect_pitches(bass_audio, sr)
    techniques = generator.detect_technique(bass_audio, sr, pitch_sequence)
    tab = generator.generate_tab(pitch_sequence, techniques)
    
    # Output
    print("\n" + tab)
    
    # Save to file
    output_file = Path(input_file).stem + "_tab.txt"
    with open(output_file, 'w') as f:
        f.write(tab)
    
    print(f"\n💾 Tab saved to: {output_file}")
    print(f"💾 Bass audio saved to: {bass_save_path}")
    print("✅ POC Complete!")


if __name__ == "__main__":
    main()
