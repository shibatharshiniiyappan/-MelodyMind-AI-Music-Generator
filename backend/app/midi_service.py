import io
import os
import tempfile

import pretty_midi

from .schemas import MelodyNote


def render_midi(
    notes: list[MelodyNote],
    tempo: int,
    instrument_program: int,
) -> bytes:
    midi = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    instrument = pretty_midi.Instrument(
        program=instrument_program,
        is_drum=False,
        name="MelodyMind generated melody",
    )
    seconds_per_beat = 60.0 / tempo

    for note in notes:
        instrument.notes.append(
            pretty_midi.Note(
                velocity=note.velocity,
                pitch=note.pitch,
                start=note.start * seconds_per_beat,
                end=(note.start + note.duration) * seconds_per_beat,
            )
        )
    midi.instruments.append(instrument)

    handle, path = tempfile.mkstemp(suffix=".mid")
    os.close(handle)
    try:
        midi.write(path)
        with open(path, "rb") as midi_file:
            return midi_file.read()
    finally:
        os.unlink(path)


def as_stream(data: bytes) -> io.BytesIO:
    return io.BytesIO(data)
