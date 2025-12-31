# Claude Code Context

## Project Overview
GPU-accelerated audio analysis package using Modal Labs. Analyzes audio files for BPM, beats, segments, key, genre, mood, instruments, and extracts high-quality vocal/instrumental stems.

## Architecture
Two-stage GPU pipeline:
- **Stage 1 (A10G)**: PyTorch - allin1 (structure), BS-RoFormer (stems), essentia (tonal)
- **Stage 2 (T4)**: TensorFlow - Essentia ML models (genre, mood, instruments, embeddings)

## Key Files
- `src/modal_audio_analysis/pipeline/app.py` - Modal app orchestration
- `src/modal_audio_analysis/pipeline/stage1.py` - PyTorch analysis (allin1, BS-RoFormer)
- `src/modal_audio_analysis/pipeline/stage2.py` - TensorFlow ML models
- `src/modal_audio_analysis/pipeline/images.py` - Modal Docker image definitions
- `src/modal_audio_analysis/cli.py` - Click CLI (analyze, batch, pricing, deploy)

## Commands
```bash
# Run analysis
uv run modal-audio-analysis analyze song.mp3 -o ./output

# Batch processing
uv run modal-audio-analysis batch ./music/ -o ./results

# Linting
uv run ruff check src/ --fix
uv run ruff format src/

# Test without installation
uvx modal-audio-analysis analyze song.mp3
```

## Test Files
Test audio files are in `~/Music/dj/ALL_MUSIC/`

## Dependencies
- Modal for serverless GPU compute
- allin1 for structure analysis (uses demucs internally)
- audio-separator for BS-RoFormer stem separation (12.9 dB SDR)
- essentia + essentia-tensorflow for audio analysis and ML

## Output Structure
```
output/
├── analysis.json      # All analysis data
├── embeddings.npy     # ML embeddings (1280-dim)
└── stems/
    ├── vocals.mp3        # Isolated vocals
    └── instrumental.mp3  # Everything else
```

## Cost
~$0.02 per 5-minute track (full analysis with stems)
