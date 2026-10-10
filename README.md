# MelodyMind — AI Music Generation with Deep Learning

MelodyMind is a working full-stack demonstration of a MIDI generation pipeline:

1. Collect legal MIDI files in a local dataset folder.
2. Extract pitches and quantized note durations with PrettyMIDI.
3. Build fixed-length training sequences.
4. Train a two-layer TensorFlow/Keras LSTM with pitch and duration outputs.
5. Autoregressively generate new note sequences from the trained model.
6. Play the sequence with a Web Audio synthesizer.
7. Export a genuine Standard MIDI File with tempo, note timing, velocity, and a
   General MIDI instrument program.

The dashboard only displays metrics returned by Keras. It does not simulate
training or fabricate loss values.

## Project structure

```text
.
├── frontend/
│   ├── index.html
│   └── src/
│       ├── App.tsx        # Studio and live training dashboard
│       ├── api.ts         # FastAPI client and shared response types
│       ├── midi.ts        # Browser-side Standard MIDI File fallback
│       └── synth.ts       # Web Audio synthesizer playback
├── backend/
│   ├── app/
│   │   ├── main.py        # FastAPI routes and background training job
│   │   ├── model.py       # LSTM, training callback, and generation
│   │   ├── preprocessing.py
│   │   ├── midi_service.py
│   │   └── schemas.py
│   ├── scripts/preprocess.py
│   ├── data/midi/         # Add your dataset here
│   ├── data/processed/    # Generated NumPy training data
│   ├── models/            # Generated Keras model and metadata
│   └── requirements.txt
├── package.json           # Figma Make-compatible frontend toolchain
└── vite.config.ts
```

## Dataset

Add `.mid` or `.midi` files to `backend/data/midi/`. Nested folders are
supported. Use MIDI files you are legally permitted to train on, such as your
own compositions or a properly licensed public dataset.

Suggested sources include the MAESTRO dataset (observe its license) or your own
exported piano MIDI. The current model reads all non-drum instruments and learns
a chronological monophonic stream of pitch and quantized duration events.

To inspect preprocessing separately:

```bash
cd backend
python scripts/preprocess.py --sequence-length 32
```

The script creates `backend/data/processed/sequences.npz` and prints the real
file, note, and sequence counts.

## Run the backend

Figma Make runs the Vite preview, but it does **not** run TensorFlow or a Python
server. Run the backend separately in a local terminal with Python 3.10 or 3.11:

```bash
cd backend
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

TensorFlow installation is large and platform-specific. On unsupported systems,
follow the official TensorFlow installation guide and then install the remaining
requirements.

Open `http://localhost:8000/docs` for the interactive API documentation.

### API

- `GET /api/status` — model state, real dataset counts, epoch, and losses
- `POST /api/train` — preprocess the MIDI folder and start LSTM training
- `POST /api/generate` — generate from the loaded trained model
- `POST /api/export-midi` — render supplied note events with PrettyMIDI
- `GET /health` — service health check

Training runs in a background thread. The frontend polls `/api/status`; an error
from preprocessing or Keras is shown verbatim in the dashboard.

## Run the frontend

In Figma Make, the Vite server is already running and hot reloads source changes.
For a standalone checkout:

```bash
pnpm install
pnpm dev
```

The default frontend URL is `http://localhost:8443`, and it expects the backend
at `http://localhost:8000`. To use another backend:

```bash
VITE_API_URL=http://127.0.0.1:9000 pnpm dev
```

If the frontend origin differs from the defaults, configure backend CORS:

```bash
MELODY_CORS_ORIGINS=http://localhost:8443,http://localhost:5173 \
  uvicorn app.main:app --reload --port 8000
```

## Workflow

1. Start the backend and frontend.
2. Add MIDI files to `backend/data/midi/`.
3. Choose epochs and batch size in the training dashboard.
4. Select **Train LSTM model** and monitor actual Keras loss and epoch progress.
5. When the status becomes **Model ready**, choose generation controls and
   select **Generate with LSTM**.
6. Play the generated note events through the browser synthesizer.
7. Export `melodymind-generation.mid`.

MIDI stores performance instructions, not recorded sound. Browser playback uses
the included Web Audio synthesizer; the exported `.mid` can use any soundfont or
instrument available in a DAW or MIDI player.

## Model notes

The model has two stacked LSTM layers. Each input timestep contains normalized
MIDI pitch and duration. It predicts a 128-class pitch distribution and a
positive duration value. Generation starts from a real dataset sequence and
samples the model's pitch distribution with the selected temperature while
feeding each prediction back into the next step.

The genre buttons are creative metadata in this baseline; the model is trained
on the complete contents of the dataset rather than falsely claiming
genre-conditioned output. Use genre-specific dataset folders and separate
models if strict genre conditioning is required.
