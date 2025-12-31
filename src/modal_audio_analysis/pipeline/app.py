"""
Modal app for audio analysis pipeline.

Two-stage GPU pipeline:
- Stage 1: PyTorch (A10G, CUDA 12.4) for allin1, demucs, essentia-traditional
- Stage 2: TensorFlow (T4, CUDA 11.8) for essentia-tensorflow ML models
"""

import modal

from modal_audio_analysis.pipeline.images import pytorch_image, tensorflow_image

app = modal.App("modal-audio-analysis")
volume = modal.Volume.from_name("audio-analysis-cache", create_if_missing=True)


# =============================================================================
# STAGE 1: PyTorch Analysis
# =============================================================================


@app.function(
    image=pytorch_image,
    gpu="A10G",
    timeout=900,
    volumes={"/cache": volume},
)
def analyze_structure(audio_bytes: bytes, filename: str, convert_stems: bool = True) -> dict:
    """Stage 1: Structure analysis using PyTorch (demucs, allin1)."""
    from modal_audio_analysis.pipeline.stage1 import run_stage1

    result = run_stage1(audio_bytes, filename, convert_stems, cache_dir="/cache")
    volume.commit()
    return result


# =============================================================================
# STAGE 2: TensorFlow ML Analysis
# =============================================================================


@app.function(
    image=tensorflow_image,
    gpu="T4",
    timeout=300,
    volumes={"/cache": volume},
)
def analyze_ml_features(audio_path: str, return_embeddings: bool = True) -> dict:
    """Stage 2: ML feature extraction using TensorFlow (GPU)."""
    from modal_audio_analysis.pipeline.stage2 import run_stage2

    return run_stage2(audio_path, return_embeddings)


# =============================================================================
# COMBINED PIPELINE
# =============================================================================


@app.function(
    image=pytorch_image,
    timeout=1200,
)
def analyze(audio_bytes: bytes, filename: str) -> dict:
    """
    Full analysis pipeline: chains Stage 1 (PyTorch) and Stage 2 (TensorFlow).

    Args:
        audio_bytes: Raw audio file bytes
        filename: Original filename

    Returns:
        dict with keys:
            - analysis: complete analysis results
            - stems_bytes: dict of stem_name -> MP3 bytes
            - embeddings_bytes: numpy .npy bytes
    """
    import time

    total_start = time.time()

    # Stage 1: Structure analysis
    print("=== STAGE 1: PyTorch Analysis (demucs, allin1) ===")
    structure_result = analyze_structure.remote(audio_bytes, filename, convert_stems=True)

    if "error" in structure_result:
        return {
            "error": structure_result["error"],
            "traceback": structure_result.get("traceback"),
        }

    stems_bytes = structure_result.pop("_stems_bytes", {})

    # Stage 2: ML analysis
    embeddings_bytes = None
    audio_path = structure_result.get("_audio_path")
    if audio_path:
        print("=== STAGE 2: TensorFlow ML Analysis ===")
        ml_result = analyze_ml_features.remote(audio_path, return_embeddings=True)

        embeddings_bytes = ml_result.pop("_embeddings_bytes", None)

        structure_result["ml_genre"] = ml_result.get("ml_genre", {})
        structure_result["ml_mood"] = ml_result.get("ml_mood", {})
        structure_result["ml_other"] = ml_result.get("ml_other", {})
        structure_result["ml_instruments"] = ml_result.get("ml_instruments", {})

        if "ml_error" in ml_result:
            structure_result["ml_error"] = ml_result["ml_error"]

        if "_timings" in ml_result:
            structure_result["_timings"]["stage2"] = ml_result["_timings"]

    structure_result.pop("_audio_path", None)

    total_time = time.time() - total_start
    structure_result["_timings"]["total"] = total_time
    print(f"\n=== TOTAL PIPELINE TIME: {total_time:.1f}s ===")

    return {
        "analysis": structure_result,
        "stems_bytes": stems_bytes,
        "embeddings_bytes": embeddings_bytes,
    }


# =============================================================================
# LOCAL ENTRY POINT
# =============================================================================


@app.local_entrypoint()
def main(audio_file: str, output_dir: str = "./analysis_output"):
    """Analyze an audio file and save results."""
    import json
    import os

    import numpy as np

    print(f"Analyzing: {audio_file}")

    with open(audio_file, "rb") as f:
        audio_bytes = f.read()

    filename = os.path.basename(audio_file)
    result = analyze.remote(audio_bytes, filename)

    if "error" in result:
        print(f"Error: {result['error']}")
        return

    os.makedirs(output_dir, exist_ok=True)

    # Save analysis JSON
    analysis_path = os.path.join(output_dir, "analysis.json")
    with open(analysis_path, "w") as f:
        json.dump(result["analysis"], f, indent=2)
    print(f"Saved: {analysis_path}")

    # Save embeddings
    if result.get("embeddings_bytes"):
        embeddings_path = os.path.join(output_dir, "embeddings.npy")
        with open(embeddings_path, "wb") as f:
            f.write(result["embeddings_bytes"])
        embeddings = np.load(embeddings_path)
        print(f"Saved: {embeddings_path} (shape: {embeddings.shape})")

    # Save stems
    stems_bytes = result.get("stems_bytes", {})
    if stems_bytes:
        stems_dir = os.path.join(output_dir, "stems")
        os.makedirs(stems_dir, exist_ok=True)
        for stem_name, stem_data in stems_bytes.items():
            stem_path = os.path.join(stems_dir, f"{stem_name}.mp3")
            with open(stem_path, "wb") as f:
                f.write(stem_data)
            print(f"Saved: {stem_path}")

    # Print summary
    analysis = result["analysis"]
    print("\n=== ANALYSIS SUMMARY ===")
    print(f"BPM: {analysis.get('structure', {}).get('bpm', 'N/A')}")
    tonal = analysis.get("tonal", {})
    print(f"Key: {tonal.get('key', 'N/A')} {tonal.get('scale', '')}")

    genres = analysis.get("ml_genre", {}).get("top_genres", [])[:3]
    if genres:
        print(f"Top Genres: {', '.join(g['genre'] for g in genres)}")

    mood = analysis.get("ml_mood", {})
    if mood:
        top_mood = max(mood.items(), key=lambda x: x[1])
        print(f"Dominant Mood: {top_mood[0]} ({top_mood[1]:.2f})")

    print(f"Danceability: {analysis.get('ml_other', {}).get('danceability', 'N/A')}")
    print(f"Voice/Instrumental: {analysis.get('ml_other', {}).get('voice_instrumental', 'N/A')}")
