from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pretty_midi

MIDI_SUFFIXES = {".mid", ".midi"}


@dataclass
class DatasetSummary:
    files: int
    notes: int
    sequences: int
    sequence_length: int
    max_duration: float


def extract_note_sequences(midi_path: Path) -> list[tuple[int, float]]:
    """Extract monophonic pitch/duration events in chronological order."""
    midi = pretty_midi.PrettyMIDI(str(midi_path))
    try:
        tempo = float(midi.estimate_tempo())
        if not np.isfinite(tempo) or tempo <= 0:
            tempo = 120.0
    except ValueError:
        tempo = 120.0

    seconds_per_beat = 60.0 / tempo
    notes = [
        note
        for instrument in midi.instruments
        if not instrument.is_drum
        for note in instrument.notes
    ]
    notes.sort(key=lambda note: (note.start, note.pitch))

    events: list[tuple[int, float]] = []
    for note in notes:
        duration_beats = (note.end - note.start) / seconds_per_beat
        quantized = round(max(0.125, duration_beats) * 8) / 8
        events.append((int(note.pitch), float(min(8.0, quantized))))
    return events


def prepare_dataset(
    source_dir: Path,
    output_path: Path,
    sequence_length: int = 32,
) -> DatasetSummary:
    midi_files = sorted(
        path
        for path in source_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in MIDI_SUFFIXES
    )
    if not midi_files:
        raise ValueError(
            f"No MIDI files found in {source_dir}. Add .mid or .midi files first."
        )

    songs = [events for path in midi_files if (events := extract_note_sequences(path))]
    note_count = sum(len(song) for song in songs)
    if note_count <= sequence_length:
        raise ValueError(
            f"Dataset has {note_count} notes; more than {sequence_length} are required."
        )

    max_duration = max(duration for song in songs for _, duration in song)
    x_values: list[np.ndarray] = []
    y_pitch: list[int] = []
    y_duration: list[float] = []

    for song in songs:
        if len(song) <= sequence_length:
            continue
        normalized = np.asarray(
            [[pitch / 127.0, duration / max_duration] for pitch, duration in song],
            dtype=np.float32,
        )
        for index in range(len(song) - sequence_length):
            x_values.append(normalized[index : index + sequence_length])
            next_pitch, next_duration = song[index + sequence_length]
            y_pitch.append(next_pitch)
            y_duration.append(next_duration / max_duration)

    if not x_values:
        raise ValueError(
            "No training sequences were created. Add longer MIDI files or reduce "
            "the sequence length."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    x_array = np.asarray(x_values, dtype=np.float32)
    np.savez_compressed(
        output_path,
        x=x_array,
        y_pitch=np.asarray(y_pitch, dtype=np.int32),
        y_duration=np.asarray(y_duration, dtype=np.float32),
        seeds=x_array,
        max_duration=np.asarray(max_duration, dtype=np.float32),
        sequence_length=np.asarray(sequence_length, dtype=np.int32),
        file_count=np.asarray(len(midi_files), dtype=np.int32),
        note_count=np.asarray(note_count, dtype=np.int32),
    )

    return DatasetSummary(
        files=len(midi_files),
        notes=note_count,
        sequences=len(x_values),
        sequence_length=sequence_length,
        max_duration=max_duration,
    )
