import { useEffect, useRef, useState } from "react";
import {
  exportMidiFromBackend,
  generateMelody,
  getModelStatus,
  startTraining,
  type MelodyNote,
  type ModelStatus,
} from "./api";
import { createMidiFile, downloadBlob } from "./midi";
import { playWithSynth, stopSynth, type SynthSession } from "./synth";

const genres = ["Classical", "Jazz", "Lo-fi", "Ambient"] as const;
type Genre = (typeof genres)[number];

const genreDetails: Record<Genre, { icon: string; description: string }> = {
  Classical: { icon: "𝄞", description: "Timeless & rich" },
  Jazz: { icon: "♩", description: "Smooth & expressive" },
  "Lo-fi": { icon: "◒", description: "Relaxed & mellow" },
  Ambient: { icon: "✧", description: "Dreamy & calm" },
};

const instruments = [
  { value: 0, label: "Acoustic Grand Piano" },
  { value: 4, label: "Electric Piano" },
  { value: 24, label: "Nylon Guitar" },
  { value: 40, label: "Violin" },
  { value: 73, label: "Flute" },
];

const offlineStatus: ModelStatus = {
  status: "offline",
  model_available: false,
  current_epoch: 0,
  total_epochs: 0,
  loss: null,
  pitch_loss: null,
  duration_loss: null,
  dataset_files: 0,
  dataset_notes: 0,
  sequences: 0,
  error: null,
};

function statusLabel(status: ModelStatus) {
  if (status.status === "offline") return "Backend offline";
  if (status.status === "preprocessing") return "Preparing MIDI dataset";
  if (status.status === "training") return "Training LSTM";
  if (status.status === "error") return "Training failed";
  if (status.model_available) return "Model ready";
  return "Awaiting training";
}

export default function App() {
  const [genre, setGenre] = useState<Genre>("Lo-fi");
  const [tempo, setTempo] = useState(90);
  const [length, setLength] = useState(16);
  const [temperature, setTemperature] = useState(0.9);
  const [instrument, setInstrument] = useState(4);
  const [melody, setMelody] = useState<MelodyNote[]>([]);
  const [playing, setPlaying] = useState(false);
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState("Connect the AI backend to begin");
  const [error, setError] = useState<string | null>(null);
  const [modelStatus, setModelStatus] =
    useState<ModelStatus>(offlineStatus);
  const [epochs, setEpochs] = useState(20);
  const [batchSize, setBatchSize] = useState(64);
  const synthRef = useRef<SynthSession | null>(null);

  async function refreshStatus(silent = false) {
    try {
      const next = await getModelStatus();
      setModelStatus(next);
      if (!silent && next.model_available) setMessage("LSTM model is ready");
    } catch {
      setModelStatus(offlineStatus);
      if (!silent) {
        setMessage("Backend offline — see README for launch commands");
      }
    }
  }

  useEffect(() => {
    void refreshStatus();
    const interval = setInterval(() => void refreshStatus(true), 2000);
    return () => {
      clearInterval(interval);
      stopSynth(synthRef.current);
    };
  }, []);

  async function handleGenerate() {
    stopPlayback();
    setBusy(true);
    setError(null);
    setMessage("Generating with the trained LSTM model…");

    try {
      const result = await generateMelody({
        length,
        temperature,
        tempo,
        instrument,
        genre,
      });
      setMelody(result.notes);
      setMessage(`Generated ${result.notes.length} notes with ${result.model}`);
    } catch (cause) {
      const nextError =
        cause instanceof Error ? cause.message : "Melody generation failed";
      setError(nextError);
      setMessage("Generation failed");
    } finally {
      setBusy(false);
    }
  }

  async function handleTrain() {
    setError(null);
    setMessage("Starting dataset preprocessing and training…");

    try {
      await startTraining({
        epochs,
        batch_size: batchSize,
        sequence_length: 32,
      });
      setMessage("Training started — progress updates automatically");
      await refreshStatus(true);
    } catch (cause) {
      const nextError =
        cause instanceof Error ? cause.message : "Could not start training";
      setError(nextError);
      setMessage("Training could not start");
    }
  }

  function stopPlayback(updateMessage = false) {
    stopSynth(synthRef.current);
    synthRef.current = null;
    setPlaying(false);
    if (updateMessage) setMessage("Playback stopped");
  }

  function handlePlayback() {
    if (playing) {
      stopPlayback(true);
      return;
    }
    if (!melody.length) {
      setError("Generate a melody with the trained model before playback.");
      return;
    }

    try {
      setError(null);
      synthRef.current = playWithSynth(
        melody,
        tempo,
        instrument,
        () => {
          synthRef.current = null;
          setPlaying(false);
          setMessage("Playback complete");
        },
      );
      setPlaying(true);
      setMessage("Playing through the Web Audio synthesizer");
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Playback failed");
    }
  }

  async function handleExport() {
    if (!melody.length) {
      setError("Generate a melody before exporting MIDI.");
      return;
    }

    setError(null);
    setMessage("Building MIDI file…");
    try {
      const blob = await exportMidiFromBackend({
        notes: melody,
        tempo,
        instrument,
      });
      downloadBlob(blob, "melodymind-generation.mid");
      setMessage("MIDI file exported");
    } catch {
      const blob = createMidiFile(melody, tempo, instrument);
      downloadBlob(blob, "melodymind-generation.mid");
      setMessage("MIDI exported locally (backend was unavailable)");
    }
  }

  const training = ["training", "preprocessing"].includes(modelStatus.status);
  const progress =
    modelStatus.total_epochs > 0
      ? (modelStatus.current_epoch / modelStatus.total_epochs) * 100
      : 0;

  return (
    <main className="app">
      <div className="shell">
        <header className="top">
          <div className="brand">
            <span className="logo" aria-hidden="true">♫</span>
            <span>Melody<span className="gradient">Mind</span></span>
          </div>
          <span className={`pill connection ${modelStatus.status}`}>
            <span className="connection-dot" />
            {statusLabel(modelStatus)}
          </span>
        </header>

        <section className="hero">
          <p className="eyebrow">DEEP LEARNING MUSIC LAB</p>
          <h1>Turn learned patterns into <span className="gradient">music.</span></h1>
          <p className="muted hero-copy">
            Train an LSTM on MIDI data, generate original note sequences, and
            export production-ready MIDI.
          </p>
        </section>

        <div className="layout">
          <section className="panel">
            <div className="section-heading">
              <span className="heading-icon" aria-hidden="true">✧</span>
              <div>
                <h2>Create a new melody</h2>
                <p className="muted">Generate from patterns learned by your model.</p>
              </div>
            </div>

            <div className="genres">
              {genres.map((item) => (
                <button
                  key={item}
                  className={`genre ${genre === item ? "active" : ""}`}
                  onClick={() => setGenre(item)}
                  type="button"
                >
                  <span className="genre-icon" aria-hidden="true">
                    {genreDetails[item].icon}
                  </span>
                  <strong>{item}</strong>
                  <span className="muted">{genreDetails[item].description}</span>
                </button>
              ))}
            </div>

            <div className="control-grid">
              <div className="field">
                <label htmlFor="instrument">MIDI instrument</label>
                <select
                  id="instrument"
                  value={instrument}
                  onChange={(event) => setInstrument(Number(event.target.value))}
                >
                  {instruments.map((item) => (
                    <option key={item.value} value={item.value}>{item.label}</option>
                  ))}
                </select>
              </div>
              <div className="field">
                <label htmlFor="temperature">
                  Creativity <strong>{temperature.toFixed(1)}</strong>
                </label>
                <input
                  id="temperature"
                  type="range"
                  min="0.2"
                  max="1.5"
                  step="0.1"
                  value={temperature}
                  onChange={(event) => setTemperature(Number(event.target.value))}
                />
              </div>
            </div>

            <div className="field">
              <label htmlFor="tempo">Tempo <strong>{tempo} BPM</strong></label>
              <input
                id="tempo"
                type="range"
                min="50"
                max="160"
                value={tempo}
                onChange={(event) => setTempo(Number(event.target.value))}
              />
              <div className="range-labels muted">
                <span>Slow and peaceful</span><span>Fast and energetic</span>
              </div>
            </div>

            <div className="field">
              <label htmlFor="length">
                Melody length <strong>{length} notes</strong>
              </label>
              <input
                id="length"
                type="range"
                min="8"
                max="64"
                step="4"
                value={length}
                onChange={(event) => setLength(Number(event.target.value))}
              />
            </div>

            <button
              className="primary generate"
              onClick={handleGenerate}
              disabled={busy || training}
            >
              {busy ? <span className="spinner" /> : <span aria-hidden="true">✦</span>}
              {busy ? "Generating…" : "Generate with LSTM"}
            </button>
          </section>

          <section className="panel composition">
            <div className="section-heading">
              <span className="heading-icon" aria-hidden="true">♫</span>
              <div>
                <h2>Your composition</h2>
                <p className="muted">
                  {instruments.find((item) => item.value === instrument)?.label} · {tempo} BPM
                </p>
              </div>
            </div>

            <div className={`wave ${playing ? "is-playing" : ""}`}>
              {melody.length ? melody.map((note, index) => (
                <div
                  key={`${note.pitch}-${index}`}
                  className="bar"
                  title={`MIDI ${note.pitch} · ${note.duration.toFixed(2)} beats`}
                  style={{
                    height: `${Math.max(20, Math.min(92, 25 + (note.pitch - 36) * 1.1))}%`,
                    animationDelay: `${index * 45}ms`,
                  }}
                />
              )) : (
                <div className="empty">
                  <span className="empty-icon" aria-hidden="true">〰</span>
                  Train or load a model, then generate a melody.
                  <br />No random fallback is used.
                </div>
              )}
            </div>

            <div className="actions">
              <button className="primary" onClick={handlePlayback}>
                {playing ? "■ Stop" : "▶ Play melody"}
              </button>
              <button className="secondary" onClick={handleExport}>
                ↓ Export .mid
              </button>
            </div>

            <div className="status" role="status">
              <span className="status-dot" /> {message}
            </div>
            {error && <div className="error-banner" role="alert">{error}</div>}

            <div className="stats">
              <div className="stat">
                <span className="muted">Generation source</span>
                <strong className="style-value">
                  {melody.length ? "LSTM model" : "—"}
                </strong>
              </div>
              <div className="stat">
                <span className="muted">Note sequence</span>
                <strong>{melody.length || "—"}</strong>
              </div>
            </div>
          </section>
        </div>

        <section className="panel training-panel">
          <div className="training-header">
            <div className="section-heading">
              <span className="heading-icon" aria-hidden="true">⌁</span>
              <div>
                <h2>Training dashboard</h2>
                <p className="muted">Live metrics reported by TensorFlow/Keras.</p>
              </div>
            </div>
            <span className={`model-badge ${modelStatus.status}`}>
              {statusLabel(modelStatus)}
            </span>
          </div>

          <div className="training-stats">
            <div className="metric">
              <span>Epoch</span>
              <strong>
                {modelStatus.total_epochs
                  ? `${modelStatus.current_epoch} / ${modelStatus.total_epochs}`
                  : "—"}
              </strong>
            </div>
            <div className="metric">
              <span>Loss</span>
              <strong>
                {modelStatus.loss === null ? "—" : modelStatus.loss.toFixed(4)}
              </strong>
            </div>
            <div className="metric">
              <span>MIDI files</span>
              <strong>{modelStatus.dataset_files || "—"}</strong>
            </div>
            <div className="metric">
              <span>Training sequences</span>
              <strong>{modelStatus.sequences || "—"}</strong>
            </div>
          </div>

          <div className="progress-track" aria-label="Training progress">
            <span style={{ width: `${progress}%` }} />
          </div>

          <div className="training-controls">
            <label>
              Epochs
              <input
                type="number"
                min="1"
                max="500"
                value={epochs}
                onChange={(event) => setEpochs(Number(event.target.value))}
              />
            </label>
            <label>
              Batch size
              <select
                value={batchSize}
                onChange={(event) => setBatchSize(Number(event.target.value))}
              >
                <option value="16">16</option>
                <option value="32">32</option>
                <option value="64">64</option>
                <option value="128">128</option>
              </select>
            </label>
            <button
              className="secondary train-button"
              onClick={handleTrain}
              disabled={training || modelStatus.status === "offline"}
            >
              {training ? <><span className="spinner" /> Training…</> : "Train LSTM model"}
            </button>
          </div>

          {modelStatus.error && (
            <div className="error-banner" role="alert">{modelStatus.error}</div>
          )}
          <p className="dataset-hint">
            Dataset location: <code>backend/data/midi/</code>. Add legal MIDI
            files there before training.
          </p>
        </section>

        <footer>AI MUSIC GENERATION · TENSORFLOW LSTM · PRETTYMIDI</footer>
      </div>
    </main>
  );
}
