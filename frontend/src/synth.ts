import type { MelodyNote } from "./api";

export type SynthSession = {
  context: AudioContext;
  finished: ReturnType<typeof setTimeout>;
};

type OscillatorShape = OscillatorType;

const instrumentShapes: Record<number, OscillatorShape> = {
  0: "triangle",
  4: "sine",
  24: "sawtooth",
  40: "triangle",
  73: "sine",
};

function midiToFrequency(pitch: number) {
  return 440 * Math.pow(2, (pitch - 69) / 12);
}

export function playWithSynth(
  notes: MelodyNote[],
  tempo: number,
  instrument: number,
  onComplete: () => void,
): SynthSession {
  const AudioContextClass =
    window.AudioContext ??
    (window as typeof window & { webkitAudioContext?: typeof AudioContext })
      .webkitAudioContext;

  if (!AudioContextClass) {
    throw new Error("Web Audio playback is not supported in this browser.");
  }

  const context = new AudioContextClass();
  const beatSeconds = 60 / tempo;
  const startAt = context.currentTime + 0.08;
  const output = context.createGain();
  output.gain.value = 0.34;
  output.connect(context.destination);

  notes.forEach((note) => {
    const noteStart = startAt + note.start * beatSeconds;
    const noteDuration = Math.max(0.08, note.duration * beatSeconds);
    const oscillator = context.createOscillator();
    const envelope = context.createGain();

    oscillator.type = instrumentShapes[instrument] || "triangle";
    oscillator.frequency.value = midiToFrequency(note.pitch);
    envelope.gain.setValueAtTime(0.0001, noteStart);
    envelope.gain.exponentialRampToValueAtTime(
      Math.max(0.03, note.velocity / 127),
      noteStart + 0.025,
    );
    envelope.gain.exponentialRampToValueAtTime(
      0.0001,
      noteStart + noteDuration,
    );
    oscillator.connect(envelope);
    envelope.connect(output);
    oscillator.start(noteStart);
    oscillator.stop(noteStart + noteDuration + 0.02);
  });

  const endBeat = Math.max(
    ...notes.map((note) => note.start + note.duration),
    0,
  );
  const finished = setTimeout(() => {
    void context.close();
    onComplete();
  }, (endBeat * beatSeconds + 0.2) * 1000);

  return { context, finished };
}

export function stopSynth(session: SynthSession | null) {
  if (!session) return;
  clearTimeout(session.finished);
  void session.context.close().catch(() => undefined);
}
