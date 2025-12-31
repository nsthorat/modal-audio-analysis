"""Output data models for audio analysis results."""

from pydantic import BaseModel


class Segment(BaseModel):
    """A segment of the audio with a label."""

    label: str
    start: float
    end: float


class StructureResult(BaseModel):
    """Structure analysis results from allin1."""

    bpm: float | None = None
    beats: list[float] = []
    downbeats: list[float] = []
    segments: list[Segment] = []


class TonalResult(BaseModel):
    """Tonal analysis results."""

    key: str | None = None
    scale: str | None = None
    key_confidence: float | None = None


class LoudnessResult(BaseModel):
    """Loudness and dynamics analysis results."""

    loudness: float | None = None
    rms: float | None = None
    dynamic_complexity: float | None = None


class SpectralResult(BaseModel):
    """Spectral analysis results."""

    centroid_mean: float | None = None
    centroid_std: float | None = None


class StemsResult(BaseModel):
    """Stem separation results."""

    bass_energy: float | None = None
    drums_energy: float | None = None
    other_energy: float | None = None
    vocals_energy: float | None = None
    bass_ratio: float | None = None
    drums_ratio: float | None = None
    other_ratio: float | None = None
    vocals_ratio: float | None = None


class GenrePrediction(BaseModel):
    """A genre prediction with probability."""

    genre: str
    probability: float


class InstrumentPrediction(BaseModel):
    """An instrument prediction with probability."""

    instrument: str
    probability: float


class MoodScores(BaseModel):
    """Mood classification scores."""

    aggressive: float | None = None
    happy: float | None = None
    relaxed: float | None = None
    sad: float | None = None


class MLResult(BaseModel):
    """ML classification results."""

    genres: list[GenrePrediction] = []
    mood: MoodScores | None = None
    instruments: list[InstrumentPrediction] = []
    danceability: float | None = None
    voice_instrumental: str | None = None
    voice_probability: float | None = None


class TimingInfo(BaseModel):
    """Timing information for the analysis."""

    allin1_demucs: float | None = None
    stems_analysis: float | None = None
    essentia_traditional: float | None = None
    stage1_total: float | None = None
    audio_load: float | None = None
    embeddings: float | None = None
    genre: float | None = None
    mood_models: float | None = None
    danceability: float | None = None
    voice_instrumental: float | None = None
    stage2_total: float | None = None
    total: float | None = None


class AnalysisResult(BaseModel):
    """Complete audio analysis result."""

    filename: str
    duration: float | None = None
    structure: StructureResult = StructureResult()
    tonal: TonalResult = TonalResult()
    loudness: LoudnessResult = LoudnessResult()
    spectral: SpectralResult = SpectralResult()
    stems: StemsResult = StemsResult()
    ml: MLResult = MLResult()
    embeddings_path: str | None = None
    timings: TimingInfo | None = None
    error: str | None = None
