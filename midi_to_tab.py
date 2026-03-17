"""
MIDI to Bass Tab Renderer
=========================
Reads a MIDI file and renders bass tablature with:
- Numbered measures
- 4 measures per row
- EADG strings with fret positions
- Notes placed at correct beat positions
- Tempo and time signature display

Usage: python midi_to_tab.py input.mid [--tracks 12,13,14] [--output output.txt]
"""

import sys
import argparse
from pathlib import Path
import pretty_midi
import math


class BassTabRenderer:
    """Renders MIDI bass notes as ASCII tablature."""

    # Standard 4-string bass tuning (MIDI note numbers for open strings)
    BASS_TUNING = {
        'G': 43,  # G2
        'D': 38,  # D2
        'A': 33,  # A1
        'E': 28,  # E1
    }
    STRING_ORDER = ['G', 'D', 'A', 'E']  # Top to bottom in tab

    def __init__(self, measures_per_row=4, subdivisions=8):
        """
        Parameters
        ----------
        measures_per_row : int
            How many measures to display per line of tab.
        subdivisions : int
            Beat grid resolution per measure.
            8 = eighth notes in 4/4 (one slot per eighth note).
        """
        self.measures_per_row = measures_per_row
        self.subdivisions = subdivisions

    def load_midi(self, midi_path, track_indices=None):
        """
        Load MIDI and extract bass notes from specified tracks.

        Returns
        -------
        notes : list of dict
            Each dict has: pitch, start, end, velocity
        tempo : float
        time_sig : tuple (numerator, denominator)
        """
        midi = pretty_midi.PrettyMIDI(midi_path)

        # Extract tempo
        tempo_changes = midi.get_tempo_changes()
        tempo = tempo_changes[1][0] if len(tempo_changes[1]) > 0 else 120.0

        # Extract time signature
        ts = midi.time_signature_changes
        if ts:
            time_sig = (ts[0].numerator, ts[0].denominator)
        else:
            time_sig = (4, 4)

        # Collect notes from specified tracks
        if track_indices is None:
            # Auto-detect bass tracks (program 32-39 = bass instruments)
            track_indices = []
            for i, inst in enumerate(midi.instruments):
                if not inst.is_drum and 32 <= inst.program <= 39:
                    track_indices.append(i)
            if not track_indices:
                print("⚠️  No bass tracks auto-detected, using all non-drum tracks")
                track_indices = [i for i, inst in enumerate(midi.instruments)
                                 if not inst.is_drum]

        all_notes = []
        for idx in track_indices:
            if idx >= len(midi.instruments):
                print(f"⚠️  Track {idx} does not exist, skipping")
                continue
            inst = midi.instruments[idx]
            print(f"  Track {idx}: program={inst.program}, "
                  f"notes={len(inst.notes)}, name='{inst.name}'")
            for note in inst.notes:
                all_notes.append({
                    'pitch': note.pitch,
                    'start': note.start,
                    'end': note.end,
                    'velocity': note.velocity,
                    'track': idx,
                })

        # Sort by start time
        all_notes.sort(key=lambda n: (n['start'], n['pitch']))
        print(f"  Total bass notes: {len(all_notes)}")

        return all_notes, tempo, time_sig

    def note_to_fret(self, midi_pitch):
        """
        Convert MIDI pitch to (string, fret) for bass guitar.

        Strategy: prefer the E string (lowest), moving to higher strings
        only when the fret would be uncomfortably high (>9). This matches
        typical bass tab conventions where players stay in lower positions.

        E.g. E2 (pitch 40) → A string fret 7  (not E string fret 12)
             B1 (pitch 35) → E string fret 7  (comfortable)
             A1 (pitch 33) → E string fret 5  (not A string fret 0)
        """
        candidates = []
        for string in reversed(self.STRING_ORDER):  # E, A, D, G
            open_pitch = self.BASS_TUNING[string]
            fret = midi_pitch - open_pitch
            if 0 <= fret <= 24:
                candidates.append((string, fret))

        if not candidates:
            return None, None

        # Prefer E string when fret <= 9, otherwise move up one string
        # This creates natural position-based playing
        string_rank = {'E': 0, 'A': 1, 'D': 2, 'G': 3}

        def score(candidate):
            string, fret = candidate
            # Strongly prefer frets 0-9
            if fret <= 9:
                return (0, string_rank[string], fret)
            else:
                return (1, string_rank[string], fret)

        return min(candidates, key=score)

    def build_measures(self, notes, tempo, time_sig):
        """
        Organize notes into measures on a beat grid.

        Returns
        -------
        measures : list of list
            Each measure is a list of slots (subdivisions).
            Each slot is a list of (string, fret) tuples (for chords).
        total_measures : int
        """
        beats_per_measure = time_sig[0]
        beat_duration = 60.0 / tempo
        measure_duration = beats_per_measure * beat_duration
        slots_per_beat = self.subdivisions // beats_per_measure

        # Determine total measures
        if notes:
            last_note_time = max(n['start'] for n in notes)
            total_measures = int(last_note_time / measure_duration) + 1
        else:
            total_measures = 0

        # Build empty grid
        measures = []
        for _ in range(total_measures):
            measure = [[] for _ in range(self.subdivisions)]
            measures.append(measure)

        # Place notes into grid
        for note in notes:
            measure_idx = int(note['start'] / measure_duration)
            if measure_idx >= total_measures:
                continue

            # Position within measure (0.0 to beats_per_measure)
            time_in_measure = note['start'] - (measure_idx * measure_duration)
            beat_position = time_in_measure / beat_duration

            # Snap to nearest subdivision slot
            slot_idx = round(beat_position * slots_per_beat)
            if slot_idx >= self.subdivisions:
                slot_idx = self.subdivisions - 1

            string, fret = self.note_to_fret(note['pitch'])
            if string is not None:
                measures[measure_idx][slot_idx].append((string, fret))

        return measures, total_measures

    def render(self, measures, total_measures, tempo, time_sig):
        """
        Render measures as ASCII bass tablature.

        Returns
        -------
        tab_text : str
        """
        lines = []

        # Header
        lines.append(f"Bass Tablature")
        lines.append(f"Tempo: ♩= {tempo:.0f}    Time Signature: "
                      f"{time_sig[0]}/{time_sig[1]}")
        lines.append("")

        # Characters per slot in a measure
        # Each slot gets 3 chars wide (enough for 2-digit fret + separator)
        slot_width = 3
        measure_width = self.subdivisions * slot_width

        # Render in rows of measures_per_row
        num_rows = math.ceil(total_measures / self.measures_per_row)

        for row in range(num_rows):
            start_m = row * self.measures_per_row
            end_m = min(start_m + self.measures_per_row, total_measures)
            row_measures = list(range(start_m, end_m))

            if not row_measures:
                break

            # Measure number line
            num_line = ""
            for m_idx in row_measures:
                label = str(m_idx + 1)
                num_line += label + " " * (measure_width + 1 - len(label))
            lines.append(num_line.rstrip())

            # String lines (G, D, A, E from top to bottom)
            for string in self.STRING_ORDER:
                string_line = ""
                for m_idx in row_measures:
                    measure = measures[m_idx] if m_idx < len(measures) else \
                        [[] for _ in range(self.subdivisions)]

                    # Build this measure for this string
                    seg = ""
                    for slot_idx in range(self.subdivisions):
                        slot_notes = measure[slot_idx]
                        # Find if this string has a note in this slot
                        fret_val = None
                        for s, f in slot_notes:
                            if s == string:
                                fret_val = f
                                break

                        if fret_val is not None:
                            fret_str = str(fret_val)
                            if len(fret_str) == 1:
                                seg += f"-{fret_str}-"
                            else:
                                seg += f"{fret_str}-"
                        else:
                            seg += "---"

                    string_line += seg + "|"

                lines.append(f"{string}|{string_line}")

            # Add empty line between rows
            lines.append("")

        return "\n".join(lines)

    def generate(self, midi_path, track_indices=None):
        """Full pipeline: load MIDI → build measures → render tab."""
        print(f"🎸 Loading MIDI: {midi_path}")
        notes, tempo, time_sig = self.load_midi(midi_path, track_indices)

        print(f"📊 Tempo: {tempo:.0f} BPM, Time Sig: {time_sig[0]}/{time_sig[1]}")
        print(f"🔨 Building measure grid (subdivisions={self.subdivisions})...")
        measures, total_measures = self.build_measures(notes, tempo, time_sig)
        print(f"   {total_measures} measures total")

        print(f"📝 Rendering tab ({self.measures_per_row} measures per row)...")
        tab = self.render(measures, total_measures, tempo, time_sig)

        return tab


def main():
    parser = argparse.ArgumentParser(description="MIDI to Bass Tab Renderer")
    parser.add_argument("midi_file", help="Path to MIDI file")
    parser.add_argument("--tracks", type=str, default=None,
                        help="Comma-separated track indices (e.g., 12,13,14). "
                             "Auto-detects bass tracks if not specified.")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Output file path (prints to stdout if not specified)")
    parser.add_argument("--measures-per-row", type=int, default=4,
                        help="Measures per row (default: 4)")
    parser.add_argument("--subdivisions", type=int, default=8,
                        help="Grid subdivisions per measure (default: 8 = eighth notes)")

    args = parser.parse_args()

    if not Path(args.midi_file).exists():
        print(f"❌ Error: File '{args.midi_file}' not found")
        sys.exit(1)

    track_indices = None
    if args.tracks:
        track_indices = [int(t.strip()) for t in args.tracks.split(",")]

    renderer = BassTabRenderer(
        measures_per_row=args.measures_per_row,
        subdivisions=args.subdivisions,
    )

    tab = renderer.generate(args.midi_file, track_indices)

    # Output
    print("\n" + "=" * 60)
    print(tab)
    print("=" * 60)

    if args.output:
        with open(args.output, "w") as f:
            f.write(tab)
        print(f"\n💾 Tab saved to: {args.output}")

    print("✅ Done!")


if __name__ == "__main__":
    main()
