import json
import threading
from pathlib import Path
from typing import Any, Callable

import numpy as np
import tensorflow as tf


class ModelStore:
    def __init__(self) -> None:
        self.model: tf.keras.Model | None = None
        self.seeds: np.ndarray | None = None
        self.max_duration = 1.0
        self.sequence_length = 32
        self.lock = threading.RLock()


store = ModelStore()


def build_model(sequence_length: int) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(sequence_length, 2), name="note_sequence")
    hidden = tf.keras.layers.LSTM(256, return_sequences=True)(inputs)
    hidden = tf.keras.layers.Dropout(0.25)(hidden)
    hidden = tf.keras.layers.LSTM(256)(hidden)
    hidden = tf.keras.layers.Dense(256, activation="relu")(hidden)
    hidden = tf.keras.layers.Dropout(0.2)(hidden)
    pitch = tf.keras.layers.Dense(128, activation="softmax", name="pitch")(hidden)
    duration = tf.keras.layers.Dense(1, activation="softplus", name="duration")(hidden)

    model = tf.keras.Model(inputs=inputs, outputs={"pitch": pitch, "duration": duration})
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss={
            "pitch": "sparse_categorical_crossentropy",
            "duration": "mse",
        },
        metrics={"pitch": ["sparse_categorical_accuracy"]},
    )
    return model


class ProgressCallback(tf.keras.callbacks.Callback):
    def __init__(self, update: Callable[[int, dict[str, float]], None]) -> None:
        super().__init__()
        self.update = update

    def on_epoch_end(self, epoch: int, logs: dict[str, float] | None = None) -> None:
        self.update(epoch + 1, logs or {})


def train_model(
    dataset_path: Path,
    model_path: Path,
    metadata_path: Path,
    epochs: int,
    batch_size: int,
    progress: Callable[[int, dict[str, float]], None],
) -> dict[str, int | float]:
    dataset = np.load(dataset_path)
    x = dataset["x"]
    y_pitch = dataset["y_pitch"]
    y_duration = dataset["y_duration"]
    sequence_length = int(dataset["sequence_length"])
    max_duration = float(dataset["max_duration"])

    model = build_model(sequence_length)
    model.fit(
        x,
        {"pitch": y_pitch, "duration": y_duration},
        epochs=epochs,
        batch_size=min(batch_size, len(x)),
        shuffle=True,
        callbacks=[ProgressCallback(progress)],
        verbose=1,
    )

    
    model_path.parent.mkdir(parents=True, exist_ok=True)

    # Save to a temporary .keras file first.
    temp_model_path = model_path.with_name("melody_lstm_temp.keras")
    if temp_model_path.exists():
        temp_model_path.unlink()

    model.save(str(temp_model_path))

    # Verify that the saved archive contains its weights.
    import zipfile

    with zipfile.ZipFile(temp_model_path, "r") as archive:
        if "config.json" not in archive.namelist():
            raise RuntimeError("Saved model is missing config.json.")
        if "model.weights.h5" not in archive.namelist():
            raise RuntimeError("Saved model is missing model.weights.h5.")

    # Confirm that Keras can reload the saved model.
    tf.keras.models.load_model(temp_model_path)

    # Replace the previous model only after verification succeeds.
    temp_model_path.replace(model_path)

    metadata_path.write_text(

        json.dumps(
            {
                "sequence_length": sequence_length,
                "max_duration": max_duration,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    with store.lock:
        store.model = model
        store.seeds = dataset["seeds"]
        store.sequence_length = sequence_length
        store.max_duration = max_duration

    return {
        "sequences": len(x),
        "files": int(dataset["file_count"]),
        "notes": int(dataset["note_count"]),
    }


def load_saved_model(
    model_path: Path,
    metadata_path: Path,
    dataset_path: Path,
) -> bool:
    if not model_path.exists() or not metadata_path.exists() or not dataset_path.exists():
        return False

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    dataset = np.load(dataset_path)
    with store.lock:
        store.model = tf.keras.models.load_model(model_path)
        store.seeds = dataset["seeds"]
        store.sequence_length = int(metadata["sequence_length"])
        store.max_duration = float(metadata["max_duration"])
    return True


def _sample_pitch(probabilities: np.ndarray, temperature: float) -> int:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    logits = np.log(np.clip(probabilities, 1e-9, 1.0)) / temperature
    logits -= np.max(logits)
    adjusted = np.exp(logits)
    adjusted /= adjusted.sum()
    return int(np.random.choice(len(adjusted), p=adjusted))


def generate_notes(length: int, temperature: float) -> list[dict[str, Any]]:
    with store.lock:
        if store.model is None or store.seeds is None or not len(store.seeds):
            raise RuntimeError("No trained model is loaded. Train the LSTM model first.")
        model = store.model
        max_duration = store.max_duration
        seed = store.seeds[np.random.randint(0, len(store.seeds))].copy()

    generated: list[dict[str, Any]] = []
    start = 0.0
    for _ in range(length):
        prediction = model.predict(seed[np.newaxis, ...], verbose=0)
        pitch = _sample_pitch(prediction["pitch"][0], temperature)
        normalized_duration = float(prediction["duration"][0][0])
        duration = round(np.clip(normalized_duration * max_duration, 0.125, 8.0) * 8) / 8
        generated.append(
            {
                "pitch": pitch,
                "duration": float(duration),
                "start": float(start),
                "velocity": 92,
            }
        )
        start += duration
        next_event = np.asarray([pitch / 127.0, duration / max_duration], dtype=np.float32)
        seed = np.concatenate([seed[1:], next_event[np.newaxis, :]], axis=0)

    return generated
