# Modal Audio Analysis

GPU-accelerated audio analysis using [Modal](https://modal.com). Extract BPM, beats, key, stems, genre, mood, and more from audio files.

## Features

- **Structure Analysis**: BPM, beats, downbeats, segments (intro, verse, chorus, etc.)
- **Stem Separation**: Vocals and instrumental (via BS-RoFormer, 12.9 dB SDR)
- **Tonal Analysis**: Key and scale detection
- **ML Classification**: Genre (400 styles), mood, danceability, instruments
- **Embeddings**: Discogs-EffNet embeddings for similarity search

## Architecture

Two-stage GPU pipeline for optimal performance:

| Stage | GPU | Libraries | Purpose |
|-------|-----|-----------|---------|
| Stage 1 | A10G | allin1, BS-RoFormer, essentia | Structure, stems, tonal |
| Stage 2 | T4 | essentia-tensorflow | ML models, embeddings |

## Installation

```bash
# Using uv (recommended)
uv add modal-audio-analysis

# Using pip
pip install modal-audio-analysis
```

### Prerequisites

1. [Modal account](https://modal.com) with API credentials
2. Python 3.11+

```bash
# Set up Modal credentials
modal token new
```

## CLI Usage

### Analyze a single file

```bash
modal-audio-analysis analyze song.mp3 -o ./output
```

**Single file output:**
```
output/
├── analysis.json     # All analysis data (JSON)
├── embeddings.npy    # ML embeddings (NumPy array, 1280-dim)
└── stems/
    ├── vocals.mp3       # Isolated vocals (BS-RoFormer, 12.9 dB SDR)
    └── instrumental.mp3 # Everything else
```

**Batch output (one folder per track):**
```
results/
├── Song One/
│   ├── analysis.json
│   ├── embeddings.npy
│   └── stems/
├── Song Two/
│   ├── analysis.json
│   ├── embeddings.npy
│   └── stems/
└── ...
```

### Batch analyze multiple files

```bash
# Analyze all audio in a folder (recursive)
modal-audio-analysis batch ./music/ -o ./results

# Analyze specific files
modal-audio-analysis batch *.mp3 -o ./results

# With concurrency control
modal-audio-analysis batch ./music/ -o ./results -j 10
```

Supported formats: `.mp3`, `.wav`, `.flac`, `.m4a`, `.ogg`, `.aac`, `.wma`

This runs files in parallel on Modal for ~4x speedup.

### Options

```bash
modal-audio-analysis analyze song.mp3 --no-stems      # Skip stem separation
modal-audio-analysis analyze song.mp3 --json-only    # Only output analysis.json
modal-audio-analysis pricing                          # Show cost estimates
```

### Run without installation (uvx)

```bash
uvx modal-audio-analysis analyze song.mp3
```

## Python SDK

### Basic usage

```python
from modal_audio_analysis.pipeline.app import analyze, app

# Read audio file
with open("song.mp3", "rb") as f:
    audio_bytes = f.read()

# Run analysis on Modal
with app.run():
    result = analyze.remote(audio_bytes, "song.mp3")

# Access results
print(f"BPM: {result['analysis']['structure']['bpm']}")
print(f"Key: {result['analysis']['tonal']['key']}")
print(f"Top genre: {result['analysis']['ml_genre']['top_genres'][0]['genre']}")

# Save stems
for stem_name, stem_bytes in result['stems_bytes'].items():
    with open(f"{stem_name}.mp3", "wb") as f:
        f.write(stem_bytes)

# Save embeddings
import numpy as np
with open("embeddings.npy", "wb") as f:
    f.write(result['embeddings_bytes'])
```

### Batch processing

```python
from modal_audio_analysis.pipeline.app import analyze, app

# Prepare inputs: list of (audio_bytes, filename) tuples
inputs = []
for path in audio_files:
    with open(path, "rb") as f:
        inputs.append((f.read(), path.name))

# Process in parallel
with app.run():
    for result in analyze.starmap(inputs):
        print(f"Analyzed: {result['analysis']['filename']}")
```

### Using the config

```python
from modal_audio_analysis.config import AnalysisConfig

# Default config (all features enabled)
config = AnalysisConfig()

# Minimal config (structure only, cheapest)
config = AnalysisConfig.minimal()

# No stems (faster)
config = AnalysisConfig.no_stems()

# Custom config
config = AnalysisConfig(
    stems={"enabled": False},
    ml={"extract_instruments": False},
)
```

## Output Schema

### analysis.json

```json
{
  "filename": "song.mp3",
  "structure": {
    "bpm": 128.0,
    "beats": [0.46, 0.93, 1.40, ...],
    "downbeats": [0.46, 2.34, ...],
    "segments": [
      {"label": "intro", "start": 0.0, "end": 16.0},
      {"label": "verse", "start": 16.0, "end": 48.0},
      ...
    ]
  },
  "tonal": {
    "key": "Am",
    "scale": "minor",
    "key_confidence": 0.85
  },
  "dynamics": {
    "loudness": 0.45,
    "rms": 0.12,
    "dynamic_complexity": 3.2
  },
  "stems": {
    "vocals_energy": 0.25,
    "instrumental_energy": 0.75,
    "vocals_ratio": 0.25,
    "instrumental_ratio": 0.75
  },
  "ml_genre": {
    "top_genres": [
      {"genre": "Electronic---Techno", "probability": 0.85},
      {"genre": "Electronic---House", "probability": 0.12},
      ...
    ]
  },
  "ml_mood": {
    "aggressive": 0.2,
    "happy": 0.7,
    "relaxed": 0.4,
    "sad": 0.1
  },
  "ml_other": {
    "danceability": 0.88,
    "voice_instrumental": "instrumental",
    "voice_probability": 0.08
  },
  "ml_instruments": {
    "instruments": [
      {"instrument": "synthesizer", "probability": 0.92},
      {"instrument": "drum machine", "probability": 0.85},
      ...
    ]
  }
}
```

## Pricing

| GPU | Per Second | Per Hour | Used For |
|-----|------------|----------|----------|
| A10G | $0.000306 | $1.10 | Stage 1 (structure, stems) |
| T4 | $0.000164 | $0.59 | Stage 2 (ML models) |

### Estimated cost per track (5 min song)

| Configuration | Stage 1 | Stage 2 | Total |
|---------------|---------|---------|-------|
| Full analysis | ~$0.014 | ~$0.005 | **~$0.02** |
| No stems | ~$0.006 | ~$0.005 | ~$0.01 |
| Structure only | ~$0.012 | - | ~$0.01 |

Batch processing with parallelization provides ~4x speedup without additional cost.

## Development

```bash
# Clone and install
git clone https://github.com/yourusername/modal-audio-analysis
cd modal-audio-analysis
uv sync

# Run linting
uv run ruff check src/
uv run ruff format src/

# Run tests
uv run pytest
```

## Libraries Used

- [allin1](https://github.com/mir-aidj/all-in-one) - Structure analysis (beats, segments)
- [audio-separator](https://github.com/karaokenerds/python-audio-separator) - Stem separation (BS-RoFormer)
- [essentia](https://essentia.upf.edu/) - Audio analysis and ML models
- [Modal](https://modal.com) - Serverless GPU compute

## License

MIT
