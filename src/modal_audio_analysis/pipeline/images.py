"""Modal image definitions for the two-stage GPU pipeline."""

from pathlib import Path

import modal

# =============================================================================
# IMAGE 1: PyTorch + allin1 + demucs (CUDA 12.4)
# =============================================================================

pytorch_image = (
    modal.Image.from_registry(
        "pytorch/pytorch:2.5.0-cuda12.4-cudnn9-runtime",
    )
    .env({"DEBIAN_FRONTEND": "noninteractive", "TZ": "UTC"})
    .apt_install("ffmpeg", "git", "build-essential")
    .run_commands(
        "pip install natten==0.17.5 --no-deps "
        "-f https://whl.natten.org/cu124/torch2.5.0/index.html",
        "pip install packaging",
    )
    .pip_install(
        "numpy<2",
        "scipy<1.14",
        "huggingface_hub",
        "tqdm",
        "allin1",
        "demucs",
        "librosa",
        "soundfile",
        "hydra-core",
        "omegaconf",
        "timm",
        "einops",
        "madmom @ git+https://github.com/CPJKU/madmom.git",
        "essentia",
        "audioread",
    )
    .add_local_file(
        str(Path(__file__).parent / "patch_allin1.py"),
        "/root/patch_allin1.py",
        copy=True,
    )
    .run_commands("python3 /root/patch_allin1.py")
)

# =============================================================================
# IMAGE 2: TensorFlow + Essentia ML models (CUDA 11.8)
# =============================================================================

tensorflow_image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("ffmpeg", "curl", "libsndfile1")
    .pip_install(
        "tensorflow[and-cuda]==2.14.0",
        "essentia-tensorflow",
        "numpy<2",
        "nvidia-cuda-runtime-cu11==11.8.89",
        "nvidia-curand-cu11==10.3.0.86",
    )
    .env(
        {
            "LD_LIBRARY_PATH": (
                "/usr/local/lib/python3.10/site-packages/nvidia/curand/lib:"
                "/usr/local/lib/python3.10/site-packages/nvidia/cublas/lib:"
                "/usr/local/lib/python3.10/site-packages/nvidia/cudnn/lib:"
                "/usr/local/lib/python3.10/site-packages/nvidia/cufft/lib:"
                "/usr/local/lib/python3.10/site-packages/nvidia/cusolver/lib:"
                "/usr/local/lib/python3.10/site-packages/nvidia/cusparse/lib:"
                "/usr/local/lib/python3.10/site-packages/nvidia/cuda_runtime/lib:"
                "/usr/local/lib/python3.10/site-packages/nvidia/nccl/lib:"
                "/usr/local/lib/python3.10/site-packages/nvidia/cuda_cupti/lib"
            ),
            "TF_FORCE_GPU_ALLOW_GROWTH": "true",
        }
    )
    .run_commands(
        "mkdir -p /models/essentia",
        # Embeddings model
        "curl -L -o /models/essentia/discogs-effnet-bs64-1.pb "
        "https://essentia.upf.edu/models/feature-extractors/discogs-effnet/"
        "discogs-effnet-bs64-1.pb",
        # Genre model
        "curl -L -o /models/essentia/genre_discogs400-discogs-effnet-1.pb "
        "https://essentia.upf.edu/models/classification-heads/genre_discogs400/"
        "genre_discogs400-discogs-effnet-1.pb",
        "curl -L -o /models/essentia/genre_discogs400-discogs-effnet-1.json "
        "https://essentia.upf.edu/models/classification-heads/genre_discogs400/"
        "genre_discogs400-discogs-effnet-1.json",
        # Mood models
        "curl -L -o /models/essentia/mood_aggressive-discogs-effnet-1.pb "
        "https://essentia.upf.edu/models/classification-heads/mood_aggressive/"
        "mood_aggressive-discogs-effnet-1.pb",
        "curl -L -o /models/essentia/mood_happy-discogs-effnet-1.pb "
        "https://essentia.upf.edu/models/classification-heads/mood_happy/"
        "mood_happy-discogs-effnet-1.pb",
        "curl -L -o /models/essentia/mood_relaxed-discogs-effnet-1.pb "
        "https://essentia.upf.edu/models/classification-heads/mood_relaxed/"
        "mood_relaxed-discogs-effnet-1.pb",
        "curl -L -o /models/essentia/mood_sad-discogs-effnet-1.pb "
        "https://essentia.upf.edu/models/classification-heads/mood_sad/"
        "mood_sad-discogs-effnet-1.pb",
        # Danceability
        "curl -L -o /models/essentia/danceability-discogs-effnet-1.pb "
        "https://essentia.upf.edu/models/classification-heads/danceability/"
        "danceability-discogs-effnet-1.pb",
        # Voice/Instrumental
        "curl -L -o /models/essentia/voice_instrumental-discogs-effnet-1.pb "
        "https://essentia.upf.edu/models/classification-heads/voice_instrumental/"
        "voice_instrumental-discogs-effnet-1.pb",
        # Instruments (MTG Jamendo)
        "curl -L -o /models/essentia/mtg_jamendo_instrument-discogs-effnet-1.pb "
        "https://essentia.upf.edu/models/classification-heads/mtg_jamendo_instrument/"
        "mtg_jamendo_instrument-discogs-effnet-1.pb",
        "curl -L -o /models/essentia/mtg_jamendo_instrument-discogs-effnet-1.json "
        "https://essentia.upf.edu/models/classification-heads/mtg_jamendo_instrument/"
        "mtg_jamendo_instrument-discogs-effnet-1.json",
    )
)
