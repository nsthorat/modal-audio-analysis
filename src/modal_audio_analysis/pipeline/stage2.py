"""
Stage 2: TensorFlow ML Analysis (Essentia TensorFlow models).

Runs on T4 GPU with CUDA 11.8.
All ML models run in parallel for maximum efficiency.
"""

import io
import json
import time
from concurrent.futures import ThreadPoolExecutor


def run_stage2(audio_path: str, return_embeddings: bool = True) -> dict:
    """
    Stage 2: ML feature extraction using TensorFlow (GPU).

    Runs Essentia TensorFlow models for genre, mood, danceability,
    voice/instrumental, and instruments - all in parallel.

    Args:
        audio_path: Path to audio file on Modal volume
        return_embeddings: If True, include raw embeddings as numpy bytes

    Returns:
        dict with ml_genre, ml_mood, ml_other, instruments, and embeddings
    """
    import essentia.standard as es
    import numpy as np

    timings = {}
    stage_start = time.time()

    features = {
        "ml_genre": {},
        "ml_mood": {},
        "ml_other": {},
        "ml_instruments": {},
        "_embeddings_bytes": None,
    }

    # Load audio at 16kHz for ML models
    try:
        print(f"Loading audio from: {audio_path}")
        t0 = time.time()
        audio_16k = es.MonoLoader(filename=audio_path, sampleRate=16000)()
        timings["audio_load"] = time.time() - t0
        print(f'Loaded {len(audio_16k)} samples in {timings["audio_load"]:.2f}s')
    except Exception as e:
        return {"error": f"Failed to load audio: {e}"}

    try:
        import tensorflow as tf
        from essentia.standard import TensorflowPredict2D, TensorflowPredictEffnetDiscogs

        gpus = tf.config.list_physical_devices("GPU")
        print(f"TensorFlow GPUs: {gpus}")

        # Compute embeddings with Discogs-EffNet (required for all other models)
        print("Loading Discogs-EffNet embedding model...")
        t0 = time.time()
        embedding_model = TensorflowPredictEffnetDiscogs(
            graphFilename="/models/essentia/discogs-effnet-bs64-1.pb",
            output="PartitionedCall:1",
        )
        embeddings = embedding_model(audio_16k)
        timings["embeddings"] = time.time() - t0
        print(f'Got embeddings shape: {embeddings.shape} in {timings["embeddings"]:.2f}s')

        if return_embeddings:
            buf = io.BytesIO()
            np.save(buf, embeddings)
            features["_embeddings_bytes"] = buf.getvalue()
            print(f'  Embeddings saved as {len(features["_embeddings_bytes"])} bytes')

        # Run all ML models in parallel
        def run_genre(emb):
            t0 = time.time()
            try:
                genre_model = TensorflowPredict2D(
                    graphFilename="/models/essentia/genre_discogs400-discogs-effnet-1.pb",
                    input="serving_default_model_Placeholder",
                    output="PartitionedCall:0",
                )
                genre_predictions = genre_model(emb)

                with open("/models/essentia/genre_discogs400-discogs-effnet-1.json") as f:
                    genre_metadata = json.load(f)
                genre_labels = genre_metadata.get("classes", [])

                avg_predictions = np.mean(genre_predictions, axis=0)
                top_indices = np.argsort(avg_predictions)[::-1][:10]
                result = {
                    "top_genres": [
                        {
                            "genre": genre_labels[i] if i < len(genre_labels) else f"genre_{i}",
                            "probability": float(avg_predictions[i]),
                        }
                        for i in top_indices
                    ]
                }
                return result, time.time() - t0
            except Exception as e:
                return {"error": str(e)}, time.time() - t0

        def run_mood(emb, mood_name: str, model_path: str):
            try:
                mood_model = TensorflowPredict2D(
                    graphFilename=model_path,
                    input="model/Placeholder",
                    output="model/Softmax",
                )
                predictions = mood_model(emb)
                return mood_name, float(np.mean(predictions[:, 1]))
            except Exception:
                return mood_name, None

        def run_danceability(emb):
            t0 = time.time()
            try:
                dance_model = TensorflowPredict2D(
                    graphFilename="/models/essentia/danceability-discogs-effnet-1.pb",
                    input="model/Placeholder",
                    output="model/Softmax",
                )
                dance_pred = dance_model(emb)
                return {"danceability": float(np.mean(dance_pred[:, 1]))}, time.time() - t0
            except Exception as e:
                return {"danceability_error": str(e)}, time.time() - t0

        def run_voice_instrumental(emb):
            t0 = time.time()
            try:
                voice_model = TensorflowPredict2D(
                    graphFilename="/models/essentia/voice_instrumental-discogs-effnet-1.pb",
                    input="model/Placeholder",
                    output="model/Softmax",
                )
                voice_pred = voice_model(emb)
                avg = float(np.mean(voice_pred[:, 1]))
                return {
                    "voice_instrumental": "voice" if avg > 0.5 else "instrumental",
                    "voice_probability": avg,
                }, time.time() - t0
            except Exception as e:
                return {"voice_error": str(e)}, time.time() - t0

        def run_instruments(emb):
            t0 = time.time()
            try:
                inst_model = TensorflowPredict2D(
                    graphFilename="/models/essentia/mtg_jamendo_instrument-discogs-effnet-1.pb",
                    input="model/Placeholder",
                    output="model/Sigmoid",
                )
                inst_predictions = inst_model(emb)

                with open("/models/essentia/mtg_jamendo_instrument-discogs-effnet-1.json") as f:
                    inst_metadata = json.load(f)
                inst_labels = inst_metadata.get("classes", [])

                avg_predictions = np.mean(inst_predictions, axis=0)
                # Return instruments with probability > 0.3
                instruments = []
                for i, prob in enumerate(avg_predictions):
                    if prob > 0.3:
                        instruments.append(
                            {
                                "instrument": (
                                    inst_labels[i] if i < len(inst_labels) else f"instrument_{i}"
                                ),
                                "probability": float(prob),
                            }
                        )
                instruments.sort(key=lambda x: x["probability"], reverse=True)
                return {"instruments": instruments[:10]}, time.time() - t0
            except Exception as e:
                return {"instruments_error": str(e)}, time.time() - t0

        # Execute all models in parallel
        print("Running ML models in parallel...")
        t0_parallel = time.time()

        mood_models = {
            "aggressive": "/models/essentia/mood_aggressive-discogs-effnet-1.pb",
            "happy": "/models/essentia/mood_happy-discogs-effnet-1.pb",
            "relaxed": "/models/essentia/mood_relaxed-discogs-effnet-1.pb",
            "sad": "/models/essentia/mood_sad-discogs-effnet-1.pb",
        }

        with ThreadPoolExecutor(max_workers=8) as executor:
            # Submit all tasks
            genre_future = executor.submit(run_genre, embeddings)
            dance_future = executor.submit(run_danceability, embeddings)
            voice_future = executor.submit(run_voice_instrumental, embeddings)
            inst_future = executor.submit(run_instruments, embeddings)

            mood_futures = {
                executor.submit(run_mood, embeddings, name, path): name
                for name, path in mood_models.items()
            }

            # Collect results
            genre_result, timings["genre"] = genre_future.result()
            features["ml_genre"] = genre_result

            dance_result, timings["danceability"] = dance_future.result()
            features["ml_other"].update(dance_result)

            voice_result, timings["voice_instrumental"] = voice_future.result()
            features["ml_other"].update(voice_result)

            inst_result, timings["instruments"] = inst_future.result()
            features["ml_instruments"] = inst_result

            for future in mood_futures:
                mood_name, mood_value = future.result()
                if mood_value is not None:
                    features["ml_mood"][mood_name] = mood_value

        timings["parallel_total"] = time.time() - t0_parallel
        print(f'  Parallel ML models: {timings["parallel_total"]:.2f}s')

        timings["stage2_total"] = time.time() - stage_start
        print("\n=== STAGE 2 TIMING SUMMARY ===")
        print(f'  audio load:         {timings.get("audio_load", 0):.2f}s')
        print(f'  embeddings:         {timings.get("embeddings", 0):.2f}s')
        print(f'  parallel ML models: {timings.get("parallel_total", 0):.2f}s')
        print(f'  TOTAL:              {timings["stage2_total"]:.2f}s')
        features["_timings"] = timings

    except Exception as e:
        import traceback

        features["ml_error"] = str(e)
        features["ml_traceback"] = traceback.format_exc()

    return features
