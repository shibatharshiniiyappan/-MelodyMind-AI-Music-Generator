export type MelodyNote = {
  pitch: number;
  duration: number;
  start: number;
  velocity: number;
};

export type ModelStatus = {
  status: "offline" | "idle" | "preprocessing" | "training" | "ready" | "error";
  model_available: boolean;
  current_epoch: number;
  total_epochs: number;
  loss: number | null;
  pitch_loss: number | null;
  duration_loss: number | null;
  dataset_files: number;
  dataset_notes: number;
  sequences: number;
  error: string | null;
};

export type GenerationResult = {
  notes: MelodyNote[];
  tempo: number;
  instrument: number;
  model: string;
};

const API_URL = (
  import.meta.env.VITE_API_URL ||
  "https://melodymind-ai-music-generator.onrender.com"
).replace(/\/+$/, "");

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const response = await fetch(`${API_URL}${path}`, {
    ...options,
    headers: {
      "Content-Type": "application/json",
      ...options?.headers,
    },
  });

  if (!response.ok) {
    let message = `Request failed (${response.status})`;
    try {
      const body = await response.json();
      message = body.detail || body.message || message;
    } catch {
      // Preserve the HTTP status message when the response is not JSON.
    }
    throw new Error(message);
  }

  return response.json() as Promise<T>;
}

export async function getModelStatus(): Promise<ModelStatus> {
  return request<ModelStatus>("/api/status");
}

export async function generateMelody(payload: {
  length: number;
  temperature: number;
  tempo: number;
  instrument: number;
  genre: string;
}): Promise<GenerationResult> {
  return request<GenerationResult>("/api/generate", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export async function startTraining(payload: {
  epochs: number;
  batch_size: number;
  sequence_length: number;
}): Promise<{ message: string }> {
  return request<{ message: string }>("/api/train", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export async function exportMidiFromBackend(payload: {
  notes: MelodyNote[];
  tempo: number;
  instrument: number;
}): Promise<Blob> {
  const response = await fetch(`${API_URL}/api/export-midi`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

  if (!response.ok) {
    let message = `MIDI export failed (${response.status})`;
    try {
      const body = await response.json();
      message = body.detail || message;
    } catch {
      // Preserve the HTTP status message when the response is not JSON.
    }
    throw new Error(message);
  }

  return response.blob();
}
