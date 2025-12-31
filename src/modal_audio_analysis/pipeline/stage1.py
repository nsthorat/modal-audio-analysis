"""
Stage 1: PyTorch Analysis (demucs, allin1, traditional Essentia).

Runs on A10G GPU with CUDA 12.4.
"""

import os
import subprocess
import tempfile
import time
import warnings


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


def convert_wav_to_mp3(wav_path: str, mp3_path: str, bitrate: str = "320k") -> bool:
    """Convert WAV to MP3 using ffmpeg."""
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", wav_path, "-codec:a", "libmp3lame", "-b:a", bitrate, mp3_path],
            capture_output=True,
            check=True,
        )
        return True
    except Exception as e:
        print(f"Failed to convert {wav_path}: {e}")
        return False


def analyze_stems(demix_path: str, convert_to_mp3: bool = True) -> tuple[dict, dict]:
    """
    Analyze energy levels of separated stems and optionally convert to MP3.

    Returns:
        tuple: (analysis_dict, stems_bytes_dict)
    """
    import numpy as np
    import soundfile as sf

    analysis = {}
    stems_bytes = {}
    stem_names = ["bass", "drums", "other", "vocals"]

    for stem in stem_names:
        stem_path = os.path.join(demix_path, f"{stem}.wav")
        if os.path.exists(stem_path):
            try:
                y, sr = sf.read(stem_path, dtype="float32")
                if len(y.shape) > 1:
                    y = np.mean(y, axis=1)
                rms = np.sqrt(np.mean(y**2))
                analysis[f"{stem}_energy"] = float(rms)

                if convert_to_mp3:
                    mp3_path = stem_path.replace(".wav", ".mp3")
                    if convert_wav_to_mp3(stem_path, mp3_path):
                        with open(mp3_path, "rb") as f:
                            stems_bytes[stem] = f.read()
                        os.unlink(mp3_path)
            except Exception as e:
                analysis[f"{stem}_error"] = str(e)

    total = sum(v for k, v in analysis.items() if k.endswith("_energy"))
    if total > 0:
        for stem in stem_names:
            key = f"{stem}_energy"
            if key in analysis:
                analysis[f"{stem}_ratio"] = analysis[key] / total

    return analysis, stems_bytes


def run_stage1(
    audio_bytes: bytes,
    filename: str,
    convert_stems: bool = True,
    cache_dir: str = "/cache",
) -> dict:
    """
    Stage 1: Structure analysis using PyTorch (demucs, allin1).

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

        print("Running allin1 structure analysis...")
        t0 = time.time()
        allin1_result = allin1.analyze(
            temp_path,
            out_dir=f"{cache_dir}/allin1_output",
            demix_dir=f"{cache_dir}/demix",
            keep_byproducts=True,
        )
        timings["allin1_demucs"] = time.time() - t0
        print(f'  allin1+demucs: {timings["allin1_demucs"]:.1f}s')

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

            t0 = time.time()
            stem_name = os.path.splitext(os.path.basename(temp_path))[0]
            demix_path = f"{cache_dir}/demix/htdemucs/{stem_name}"
            if os.path.exists(demix_path):
                stem_analysis, stems_bytes = analyze_stems(demix_path, convert_to_mp3=convert_stems)
                result["stems"] = stem_analysis
                result["_stems_bytes"] = stems_bytes
                print(f"  Converted {len(stems_bytes)} stems to MP3")
            timings["stems_analysis"] = time.time() - t0
            print(f'  stems analysis + MP3 conversion: {timings["stems_analysis"]:.1f}s')

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
        print(f'  allin1+demucs:        {timings.get("allin1_demucs", 0):.1f}s')
        print(f'  stems + MP3:          {timings.get("stems_analysis", 0):.1f}s')
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
