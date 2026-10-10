from pydantic import BaseModel, Field


class MelodyNote(BaseModel):
    pitch: int = Field(ge=0, le=127)
    duration: float = Field(gt=0, le=16)
    start: float = Field(ge=0)
    velocity: int = Field(default=92, ge=1, le=127)


class GenerateRequest(BaseModel):
    length: int = Field(default=32, ge=8, le=256)
    temperature: float = Field(default=0.9, ge=0.1, le=2.0)
    tempo: int = Field(default=90, ge=30, le=240)
    instrument: int = Field(default=0, ge=0, le=127)
    genre: str = Field(default="Lo-fi", max_length=40)


class GenerateResponse(BaseModel):
    notes: list[MelodyNote]
    tempo: int
    instrument: int
    model: str = "TensorFlow LSTM"


class TrainRequest(BaseModel):
    epochs: int = Field(default=20, ge=1, le=500)
    batch_size: int = Field(default=64, ge=1, le=512)
    sequence_length: int = Field(default=32, ge=8, le=128)


class ExportRequest(BaseModel):
    notes: list[MelodyNote] = Field(min_length=1)
    tempo: int = Field(default=90, ge=30, le=240)
    instrument: int = Field(default=0, ge=0, le=127)
