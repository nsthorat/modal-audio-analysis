"""Modal pipeline for audio analysis."""

from modal_audio_analysis.pipeline.app import analyze, app

__all__ = ["app", "analyze"]
