"""Configuration models for audio analysis pipeline."""

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class StemModel(str, Enum):
    """Available stem separation models."""

    HTDEMUCS = "htdemucs"
    HTDEMUCS_FT = "htdemucs_ft"  # 4x slower but higher quality
    HTDEMUCS_6S = "htdemucs_6s"  # 6 stems (adds guitar + piano)


class EmbeddingModel(str, Enum):
    """Available embedding models."""

    DISCOGS_EFFNET = "discogs-effnet"


class GenreModel(str, Enum):
    """Available genre classification models."""

    DISCOGS400 = "discogs400"


class GPUConfig(BaseModel):
    """GPU configuration for each pipeline stage."""

    stage1_gpu: Literal["a10g", "l4", "t4"] = "a10g"
    stage2_gpu: Literal["t4", "l4", "a10g"] = "t4"


class StructureConfig(BaseModel):
    """Configuration for structure analysis (allin1)."""

    enabled: bool = True
    extract_beats: bool = True
    extract_downbeats: bool = True
    extract_segments: bool = True


class StemsConfig(BaseModel):
    """Configuration for stem separation (demucs)."""

    enabled: bool = True
    model: StemModel = StemModel.HTDEMUCS
    output_format: Literal["mp3", "wav"] = "mp3"
    mp3_bitrate: str = "320k"


class TonalConfig(BaseModel):
    """Configuration for tonal analysis."""

    enabled: bool = True
    extract_key: bool = True


class LoudnessConfig(BaseModel):
    """Configuration for loudness analysis."""

    enabled: bool = True
    extract_loudness: bool = True
    extract_dynamic_complexity: bool = True


class MLConfig(BaseModel):
    """Configuration for ML classification models."""

    enabled: bool = True
    extract_genre: bool = True
    genre_model: GenreModel = GenreModel.DISCOGS400
    extract_mood: bool = True
    extract_danceability: bool = True
    extract_voice_instrumental: bool = True
    extract_instruments: bool = True


class EmbeddingsConfig(BaseModel):
    """Configuration for audio embeddings."""

    enabled: bool = True
    model: EmbeddingModel = EmbeddingModel.DISCOGS_EFFNET
    save_to_file: bool = True


class AnalysisConfig(BaseModel):
    """Main configuration for audio analysis pipeline."""

    gpu: GPUConfig = Field(default_factory=GPUConfig)
    structure: StructureConfig = Field(default_factory=StructureConfig)
    stems: StemsConfig = Field(default_factory=StemsConfig)
    tonal: TonalConfig = Field(default_factory=TonalConfig)
    loudness: LoudnessConfig = Field(default_factory=LoudnessConfig)
    ml: MLConfig = Field(default_factory=MLConfig)
    embeddings: EmbeddingsConfig = Field(default_factory=EmbeddingsConfig)
    output_dir: str = "./analysis_output"

    @classmethod
    def minimal(cls) -> "AnalysisConfig":
        """Minimal config for fastest/cheapest analysis (structure only)."""
        return cls(
            structure=StructureConfig(enabled=True),
            stems=StemsConfig(enabled=False),
            tonal=TonalConfig(enabled=False),
            loudness=LoudnessConfig(enabled=False),
            ml=MLConfig(enabled=False),
            embeddings=EmbeddingsConfig(enabled=False),
        )

    @classmethod
    def no_stems(cls) -> "AnalysisConfig":
        """Config without stem separation (faster, cheaper)."""
        return cls(stems=StemsConfig(enabled=False))

    @classmethod
    def full(cls) -> "AnalysisConfig":
        """Full config for comprehensive analysis."""
        return cls()
