import type { MelodyNote } from "./api";

const PPQ = 480;

function uint16(value: number): number[] {
  return [(value >> 8) & 0xff, value & 0xff];
}

function uint32(value: number): number[] {
  return [
    (value >> 24) & 0xff,
    (value >> 16) & 0xff,
    (value >> 8) & 0xff,
    value & 0xff,
  ];
}

function variableLength(value: number): number[] {
  let buffer = value & 0x7f;
  const bytes: number[] = [];

  while ((value >>= 7)) {
    buffer <<= 8;
    buffer |= (value & 0x7f) | 0x80;
  }

  while (true) {
    bytes.push(buffer & 0xff);
    if (buffer & 0x80) buffer >>= 8;
    else break;
  }

  return bytes;
}

export function createMidiFile(
  notes: MelodyNote[],
  tempo: number,
  instrument: number,
): Blob {
  const microsecondsPerBeat = Math.round(60_000_000 / tempo);
  const track: number[] = [
    0x00,
    0xff,
    0x51,
    0x03,
    (microsecondsPerBeat >> 16) & 0xff,
    (microsecondsPerBeat >> 8) & 0xff,
    microsecondsPerBeat & 0xff,
    0x00,
    0xc0,
    Math.max(0, Math.min(127, instrument)),
  ];

  const events = notes
    .flatMap((note) => {
      const start = Math.round(note.start * PPQ);
      const end = Math.round((note.start + note.duration) * PPQ);
      const pitch = Math.max(0, Math.min(127, Math.round(note.pitch)));
      const velocity = Math.max(1, Math.min(127, Math.round(note.velocity)));
      return [
        { tick: start, priority: 1, data: [0x90, pitch, velocity] },
        { tick: end, priority: 0, data: [0x80, pitch, 0x00] },
      ];
    })
    .sort((a, b) => a.tick - b.tick || a.priority - b.priority);

  let previousTick = 0;
  events.forEach((event) => {
    track.push(...variableLength(event.tick - previousTick), ...event.data);
    previousTick = event.tick;
  });
  track.push(0x00, 0xff, 0x2f, 0x00);

  const header = [
    ...new TextEncoder().encode("MThd"),
    ...uint32(6),
    ...uint16(0),
    ...uint16(1),
    ...uint16(PPQ),
  ];
  const trackChunk = [
    ...new TextEncoder().encode("MTrk"),
    ...uint32(track.length),
    ...track,
  ];

  return new Blob([new Uint8Array([...header, ...trackChunk])], {
    type: "audio/midi",
  });
}

export function downloadBlob(blob: Blob, filename: string) {
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 0);
}
