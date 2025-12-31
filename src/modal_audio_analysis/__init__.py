"""Modal Audio Analysis - GPU-accelerated audio analysis using Modal Labs."""

from modal_audio_analysis.config import AnalysisConfig
from modal_audio_analysis.models import AnalysisResult

__version__ = "0.1.0"
__all__ = ["AnalysisConfig", "AnalysisResult", "__version__"]
