import os
import threading
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from .midi_service import as_stream, render_midi
from .model import generate_notes, load_saved_model, store, train_model
from .preprocessing import prepare_dataset
from .schemas import ExportRequest, GenerateRequest, GenerateResponse, TrainRequest

BASE_DIR = Path(__file__).resolve().parents[1]
MIDI_DIR = Path(os.getenv("MELODY_MIDI_DIR", BASE_DIR / "data" / "midi"))
DATASET_PATH = BASE_DIR / "data" / "processed" / "sequences.npz"
MODEL_PATH = BASE_DIR / "models" / "melody_lstm.keras"
METADATA_PATH = BASE_DIR / "models" / "metadata.json"

app = FastAPI(
    title="MelodyMind AI API",
    version="1.0.0",
    description="LSTM MIDI preprocessing, training, generation, and export.",
)
app.add_middleware(
    CORSMiddleware,
allow_origins=os.getenv(
    "MELODY_CORS_ORIGINS",
    "https://melodymind-frontend.onrender.com,http://localhost:8443,http://localhost:5173",
).split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

status_lock = threading.RLock()
training_status: dict[str, Any] = {
    "status": "idle",
    "model_available": False,
    "current_epoch": 0,
    "total_epochs": 0,
    "loss": None,
    "pitch_loss": None,
    "duration_loss": None,
    "dataset_files": 0,
    "dataset_notes": 0,
    "sequences": 0,
    "error": None,
}


def update_status(**values: Any) -> None:
    with status_lock:
        training_status.update(values)


def train_in_background(request: TrainRequest) -> None:
    try:
        update_status(
            status="preprocessing",
            current_epoch=0,
            total_epochs=request.epochs,
            loss=None,
            pitch_loss=None,
            duration_loss=None,
            error=None,
        )
        summary = prepare_dataset(
            MIDI_DIR,
            DATASET_PATH,
            sequence_length=request.sequence_length,
        )
        update_status(
            status="training",
            dataset_files=summary.files,
            dataset_notes=summary.notes,
            sequences=summary.sequences,
        )

        def report(epoch: int, logs: dict[str, float]) -> None:
            update_status(
                current_epoch=epoch,
                loss=float(logs["loss"]) if "loss" in logs else None,
                pitch_loss=float(logs["pitch_loss"]) if "pitch_loss" in logs else None,
                duration_loss=(
                    float(logs["duration_loss"]) if "duration_loss" in logs else None
                ),
            )

        result = train_model(
            DATASET_PATH,
            MODEL_PATH,
            METADATA_PATH,
            epochs=request.epochs,
            batch_size=request.batch_size,
            progress=report,
        )
        update_status(
            status="ready",
            model_available=True,
            current_epoch=request.epochs,
            dataset_files=result["files"],
            dataset_notes=result["notes"],
            sequences=result["sequences"],
        )
    except Exception as exc:
        update_status(status="error", error=str(exc))


@app.on_event("startup")
def load_artifacts() -> None:
    try:
        available = load_saved_model(MODEL_PATH, METADATA_PATH, DATASET_PATH)
        update_status(
            status="ready" if available else "idle",
            model_available=available,
        )
    except Exception as exc:
        update_status(status="error", model_available=False, error=str(exc))


@app.get("/api/status")
def get_status() -> dict[str, Any]:
    with status_lock:
        return dict(training_status)


@app.post("/api/train", status_code=202)
def start_training(request: TrainRequest) -> dict[str, str]:
    with status_lock:
        if training_status["status"] in {"preprocessing", "training"}:
            raise HTTPException(status_code=409, detail="Training is already running.")

    worker = threading.Thread(
        target=train_in_background,
        args=(request,),
        daemon=True,
        name="melodymind-training",
    )
    worker.start()
    return {"message": "Training started."}


@app.post("/api/generate", response_model=GenerateResponse)
def generate(request: GenerateRequest) -> GenerateResponse:
    try:
        notes = generate_notes(request.length, request.temperature)
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Generation failed: {exc}") from exc

    return GenerateResponse(
        notes=notes,
        tempo=request.tempo,
        instrument=request.instrument,
    )


@app.post("/api/export-midi")
def export_midi(request: ExportRequest) -> StreamingResponse:
    try:
        data = render_midi(request.notes, request.tempo, request.instrument)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"MIDI export failed: {exc}") from exc

    return StreamingResponse(
        as_stream(data),
        media_type="audio/midi",
        headers={
            "Content-Disposition": 'attachment; filename="melodymind-generation.mid"'
        },
    )


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
