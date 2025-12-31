"""
Stage 1: PyTorch Analysis (BS-RoFormer, allin1, traditional Essentia).

Runs on A10G GPU with CUDA 12.4.
Uses BS-RoFormer for high-quality vocal/instrumental separation (12.9 dB SDR).
"""

import os
import tempfile
import time
import warnings


def separate_stems_roformer(audio_path: str, output_dir: str) -> tuple[dict, dict]:
    """
    Separate vocals/instrumental using BS-RoFormer.

    Returns:
        tuple: (analysis_dict, stems_bytes_dict)
    """
    import numpy as np
    from audio_separator.separator import Separator

    separator = Separator(
        output_dir=output_dir,
        output_format="mp3",
    )

    # Load BS-RoFormer model (12.9 dB SDR - best for vocals)
    separator.load_model(model_filename="model_bs_roformer_ep_317_sdr_12.9755.ckpt")

    output_files = separator.separate(audio_path)

    analysis = {}
    stems_bytes = {}

    for output_file in output_files:
        stem_name = "vocals" if "Vocals" in output_file else "instrumental"

        full_path = os.path.join(output_dir, os.path.basename(output_file))
        if not os.path.exists(full_path):
            full_path = output_file

        if os.path.exists(full_path):
            # Read for energy analysis (mp3)
            try:
                import librosa

                y, sr = librosa.load(full_path, sr=None, mono=True)
                rms = float(np.sqrt(np.mean(y**2)))
                analysis[f"{stem_name}_energy"] = rms
            except Exception:
                pass

            # Read bytes
            with open(full_path, "rb") as f:
                stems_bytes[stem_name] = f.read()

    # Calculate ratios
    total = sum(v for k, v in analysis.items() if k.endswith("_energy"))
    if total > 0:
        for stem_name in ["vocals", "instrumental"]:
            key = f"{stem_name}_energy"
            if key in analysis:
                analysis[f"{stem_name}_ratio"] = analysis[key] / total

    return analysis, stems_bytes


def extract_traditional_features(audio_path: str) -> dict:
    """Extract features using traditional Essentia algorithms (no TensorFlow)."""
    import essentia.standard as es
    import numpy as np

    features = {
        "tonal": {},
        "dynamics": {},
        "spectral": {},
    }

    try:
        audio_44k = es.MonoLoader(filename=audio_path, sampleRate=44100)()
    except Exception as e:
        return {"error": f"Failed to load audio: {e}"}

    # Key detection
    try:
        key_extractor = es.KeyExtractor()
        key, scale, strength = key_extractor(audio_44k)
        features["tonal"]["key"] = key
        features["tonal"]["scale"] = scale
        features["tonal"]["key_confidence"] = float(strength)
    except Exception as e:
        features["tonal"]["error"] = str(e)

    # Dynamics
    try:
        features["dynamics"]["loudness"] = float(es.Loudness()(audio_44k))
        features["dynamics"]["rms"] = float(es.RMS()(audio_44k))
        features["dynamics"]["dynamic_complexity"] = float(es.DynamicComplexity()(audio_44k)[0])
    except Exception as e:
        features["dynamics"]["error"] = str(e)

    # Spectral features (summarized)
    try:
        frame_size = 2048
        hop_size = 1024
        spectrum = es.Spectrum(size=frame_size)
        centroid = es.Centroid(range=22050)

        centroids = []
        for frame in es.FrameGenerator(audio_44k, frameSize=frame_size, hopSize=hop_size):
            spec = spectrum(frame)
            centroids.append(centroid(spec))

        features["spectral"]["centroid_mean"] = float(np.mean(centroids))
        features["spectral"]["centroid_std"] = float(np.std(centroids))
    except Exception as e:
        features["spectral"]["error"] = str(e)

    return features


def run_stage1(
    audio_bytes: bytes,
    filename: str,
    separate_stems: bool = True,
    cache_dir: str = "/cache",
) -> dict:
    """
    Stage 1: Structure analysis using PyTorch (allin1, BS-RoFormer).

    Returns structure features, stem MP3 bytes, and audio path for Stage 2.
    """
    import shutil

    warnings.filterwarnings("ignore")

    os.environ["TORCH_HOME"] = f"{cache_dir}/torch"
    os.environ["HF_HOME"] = f"{cache_dir}/huggingface"

    timings = {}
    stage_start = time.time()

    suffix = os.path.splitext(filename)[1] or ".mp3"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        f.write(audio_bytes)
        temp_path = f.name

    result = {
        "filename": filename,
        "structure": {},
        "tonal": {},
        "dynamics": {},
        "spectral": {},
        "stems": {},
        "_stems_bytes": {},
    }

    try:
        import allin1
        import torch

        print(f"GPU: {torch.cuda.get_device_name(0)}")

        # Run allin1 for structure analysis (no demucs byproducts needed)
        print("Running allin1 structure analysis...")
        t0 = time.time()
        allin1_result = allin1.analyze(
            temp_path,
            out_dir=f"{cache_dir}/allin1_output",
            demix_dir=f"{cache_dir}/demix",
            keep_byproducts=False,
        )
        timings["allin1"] = time.time() - t0
        print(f'  allin1: {timings["allin1"]:.1f}s')

        if allin1_result and allin1_result.bpm:
            result["structure"] = {
                "bpm": float(allin1_result.bpm),
                "beats": [float(b) for b in allin1_result.beats] if allin1_result.beats else [],
                "downbeats": (
                    [float(d) for d in allin1_result.downbeats] if allin1_result.downbeats else []
                ),
                "segments": (
                    [
                        {
                            "label": seg.label,
                            "start": float(seg.start),
                            "end": float(seg.end),
                        }
                        for seg in allin1_result.segments
                    ]
                    if allin1_result.segments
                    else []
                ),
            }

        # Run BS-RoFormer for high-quality stem separation
        if separate_stems:
            print("Running BS-RoFormer stem separation...")
            t0 = time.time()
            stems_dir = tempfile.mkdtemp()
            try:
                stem_analysis, stems_bytes = separate_stems_roformer(temp_path, stems_dir)
                result["stems"] = stem_analysis
                result["_stems_bytes"] = stems_bytes
                timings["roformer"] = time.time() - t0
                print(f'  BS-RoFormer: {timings["roformer"]:.1f}s ({len(stems_bytes)} stems)')
            finally:
                shutil.rmtree(stems_dir, ignore_errors=True)

        print("Extracting traditional Essentia features...")
        t0 = time.time()
        essentia_features = extract_traditional_features(temp_path)
        timings["essentia_traditional"] = time.time() - t0
        print(f'  traditional essentia: {timings["essentia_traditional"]:.1f}s')
        result["tonal"] = essentia_features.get("tonal", {})
        result["dynamics"] = essentia_features.get("dynamics", {})
        result["spectral"] = essentia_features.get("spectral", {})

        volume_audio_path = f"{cache_dir}/audio_for_ml/{filename}"
        os.makedirs(os.path.dirname(volume_audio_path), exist_ok=True)
        shutil.copy(temp_path, volume_audio_path)
        result["_audio_path"] = volume_audio_path
        print(f"Saved audio to volume: {volume_audio_path}")

        timings["stage1_total"] = time.time() - stage_start
        print("\n=== STAGE 1 TIMING SUMMARY ===")
        print(f'  allin1:               {timings.get("allin1", 0):.1f}s')
        print(f'  BS-RoFormer:          {timings.get("roformer", 0):.1f}s')
        print(f'  traditional essentia: {timings.get("essentia_traditional", 0):.1f}s')
        print(f'  TOTAL:                {timings["stage1_total"]:.1f}s')
        result["_timings"] = timings

    except Exception as e:
        import traceback

        result["error"] = str(e)
        result["traceback"] = traceback.format_exc()
    finally:
        os.unlink(temp_path)

    return result
