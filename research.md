# Modal Audio Analysis - Research

Comprehensive research on audio analysis libraries and features for a Modal-based music analysis pipeline.

## Table of Contents

1. [Current Implementation](#current-implementation)
2. [allin1 - Structure Analysis](#allin1---structure-analysis)
3. [Demucs - Stem Separation](#demucs---stem-separation)
4. [audio-separator - Advanced Stem Separation](#audio-separator---advanced-stem-separation)
5. [Essentia Traditional - DSP Algorithms](#essentia-traditional---dsp-algorithms)
6. [Essentia TensorFlow - ML Models](#essentia-tensorflow---ml-models)
7. [Madmom - Beat/Music Analysis](#madmom---beatmusic-analysis)
8. [Librosa - Audio Analysis](#librosa---audio-analysis)
9. [Feature Priority Recommendations](#feature-priority-recommendations)

---

## Current Implementation

The existing playlist-manager project uses a two-stage Modal GPU pipeline:

### Stage 1 - PyTorch on A10G (CUDA 12.4)
- **allin1**: BPM, beats, downbeats, segment detection (INTRO, VERSE, CHORUS, BRIDGE, OUTRO)
- **demucs**: Stem separation → bass, drums, vocals, other (saved as MP3)
- **Essentia traditional**: Key detection, loudness, dynamics, spectral features

### Stage 2 - TensorFlow on T4 (CUDA 11.8)
- **Discogs-EffNet embeddings**: 200-dim vectors for similarity search
- **Genre classification**: 400 Discogs styles
- **Mood models**: aggressive, happy, relaxed, sad (4 separate models)
- **Danceability**
- **Voice/instrumental** classification

### Current Output Per Track
```
analysis/[spotify_id] Artist - Track/
├── stems/
│   ├── bass.mp3
│   ├── drums.mp3
│   ├── vocals.mp3
│   └── other.mp3
├── embeddings.npy      # 200-dim Discogs-EffNet
└── analysis.json       # All features
```

---

## allin1 - Structure Analysis

**Repository**: [github.com/mir-aidj/all-in-one](https://github.com/mir-aidj/all-in-one)
**Paper**: [arXiv:2307.16425](https://arxiv.org/abs/2307.16425)

### Currently Using
- BPM
- Beats (timestamps)
- Downbeats
- Segments (with labels)

### Available But Not Using

| Feature | Description |
|---------|-------------|
| `include_embeddings=True` | 24-dim latent vectors per stem (bass/drums/vocals/other), shape: `[4, time_steps, 24]` |
| `include_activations=True` | Raw frame-level sigmoid outputs (100 FPS) for beat/downbeat/segment/label |
| `beat_positions` | Metrical position (1, 2, 3, 4) for each beat |
| Individual fold models | Run `harmonix-fold0` through `harmonix-fold7` for uncertainty estimation |
| `visualize` | Built-in matplotlib visualization |
| `sonify` | Mix metronome clicks into audio |

### Segment Labels (10 Classes)
Trained on Harmonix Set dataset:
- `start`, `end`
- `intro`, `outro`
- `verse`, `chorus`
- `bridge`, `break`
- `inst` (instrumental), `solo`

### Configuration Options for `analyze()`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `paths` | PathLike/List | required | Audio file(s) to process |
| `out_dir` | PathLike | None | Save results as JSON |
| `model` | str | `'harmonix-all'` | Model to use (ensemble of 8 folds) |
| `device` | str | auto | `'cuda'` or `'cpu'` |
| `include_activations` | bool | False | Include raw frame-level activations |
| `include_embeddings` | bool | False | Include frame-level embeddings |
| `demix_dir` | PathLike | `'./demix'` | Source-separated audio location |
| `keep_byproducts` | bool | False | Retain demixed audio/spectrograms |

### Technical Notes
- Uses Demucs internally for source separation into 4 stems
- Model uses dilated neighborhood attentions for long-term dependencies
- Multi-task learning: beat/downbeat/segment tasks mutually benefit each other
- Recommended to convert MP3 to WAV first to avoid 20-40ms offset variations

---

## Demucs - Stem Separation

**Repository**: [github.com/facebookresearch/demucs](https://github.com/facebookresearch/demucs)
**Maintainer**: Meta/Facebook Research (⚠️ No longer actively maintained)

### Currently Using
- `htdemucs` model (4 stems: bass, drums, vocals, other)

### Available Models

| Model | Description | SDR (dB) |
|-------|-------------|----------|
| **htdemucs** | Hybrid Transformer Demucs v4 (default) | 9.0 |
| **htdemucs_ft** | Fine-tuned version, 4x slower but better quality | 9.2 |
| **htdemucs_6s** | 6-source model (adds guitar + piano) | - |
| **hdemucs_mmi** | Hybrid Demucs v3 | 7.7 |
| **mdx** | Won MDX Challenge Track A | - |
| **mdx_extra** | Ranked 2nd MDX Track B | - |

### Stems Available

**4-stem models**: drums, bass, vocals, other

**6-stem model (htdemucs_6s)**:
- drums, bass, vocals, other
- **guitar** (acceptable quality)
- **piano** (quality is poor - known artifacts)

### Quality Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--shifts` | Random time-shift averaging (shift trick) | 1 |
| `--segment` | Chunk size in seconds | ~10s |
| `--overlap` | Overlap between chunks | 0.25 |

### Python API

```python
from demucs.api import Separator

separator = Separator(
    model="htdemucs_ft",
    device="cuda",
    shifts=2,
    overlap=0.25,
)

original, stems = separator.separate_audio_file("track.mp3")
# stems: {"drums": tensor, "bass": tensor, "vocals": tensor, "other": tensor}
```

### Performance
- GPU (RTX 3060 Ti): ~15 seconds for 6-minute track
- CPU: ~3 minutes for 6-minute track
- `htdemucs_ft`: 4x slower than `htdemucs`

---

## audio-separator - Advanced Stem Separation

**Repository**: [github.com/nomadkaraoke/python-audio-separator](https://github.com/nomadkaraoke/python-audio-separator)
**Maintainer**: Andrew Beveridge (@beveradb) - actively maintained
**PyPI**: [audio-separator](https://pypi.org/project/audio-separator/)

### Why Use This?

Wraps Ultimate Vocal Remover (UVR) models, providing access to newer/better models:

| Model | Vocals SDR | Notes |
|-------|-----------|-------|
| **BS-RoFormer** | 12.9 dB | SOTA for vocals (won SDX'23) |
| **Mel-Band RoFormer** | 13.3 dB | Best for vocals (mel-scale) |
| **htdemucs_ft** (current) | 8.3 dB | Multi-stem but lower quality |

**RoFormer models are ~4-5 dB better** for vocal separation than demucs.

### Supported Architectures

1. **BS-RoFormer** - Band-split with rotary embeddings (SOTA)
2. **Mel-Band RoFormer** - Mel-scale frequency mapping (best for vocals)
3. **Demucs v4** - Multi-stem separation
4. **MDX-Net** - Karaoke-style separation
5. **VR Arch** - Denoising, echo removal

### Model Performance Comparison (SDR on MUSDB18HQ)

| Model | Vocals SDR | Instrumental SDR | Multi-stem |
|-------|-----------|-----------------|------------|
| **Mel-RoFormer (extra data)** | 13.29 dB | ~18.0 dB | No |
| **BS-RoFormer (extra data)** | 12.82 dB | 18.20 dB | No |
| **BS-RoFormer (MUSDB only)** | 11.49 dB | 17.0 dB | No |
| **MDX23C (8K FFT v2)** | 10.36 dB | 16.66 dB | No |
| **htdemucs_ft** | 8.33 dB | 14.63 dB | Yes (4 stems) |

### Python API

```python
from audio_separator.separator import Separator

separator = Separator()
separator.load_model(model_filename='model_bs_roformer_ep_317_sdr_12.9755.ckpt')
output_files = separator.separate('song.mp3')
# -> ['song_(Vocals).wav', 'song_(Instrumental).wav']
```

### Modal Support Built-In

```bash
pip install audio-separator[gpu] modal
modal deploy audio_separator/remote/deploy_modal.py
```

### Recommendation

Use both:
- **Demucs** for 4-stem separation (bass/drums/vocals/other)
- **BS-RoFormer** (via audio-separator) for higher-quality vocal isolation

---

## Essentia Traditional - DSP Algorithms

**Website**: [essentia.upf.edu](https://essentia.upf.edu/)
**Documentation**: [Algorithms Reference](https://essentia.upf.edu/algorithms_reference.html)

### Currently Using
- `MonoLoader` - audio loading
- `KeyExtractor` - key and scale detection
- `Loudness` - Steven's power law loudness
- `RMS` - root mean square energy
- `DynamicComplexity` - loudness variation
- `Spectrum` - magnitude spectrum
- `Centroid` - spectral centroid

### High Priority Additions

#### Loudness & Dynamics

| Algorithm | Description |
|-----------|-------------|
| `LoudnessEBUR128` | Professional LUFS measurement (momentary, short-term, integrated, loudness range) |
| `ReplayGain` | Track normalization value |
| `TruePeakDetector` | Inter-sample peak detection (clipping) |
| `Intensity` | Relaxed/moderate/aggressive classification |

#### Rhythm Features

| Algorithm | Description |
|-----------|-------------|
| `OnsetRate` | Onsets per second (rhythmic density) |
| `OnsetDetection` | Multiple methods: HFC, Complex, Flux, Melflux, RMS |
| `BpmHistogramDescriptors` | BPM stability (first/second peak, spread, weight) |
| `BpmRubato` | Tempo variation/rubato regions |
| `Meter` | Time signature estimation (3/4 vs 4/4) |
| `BeatsLoudness` | Energy at beat positions per frequency band |

#### Tonal Features

| Algorithm | Description |
|-----------|-------------|
| `HPCP` | Harmonic Pitch Class Profile (12+ dimensional) |
| `ChordsDetection` | Chord estimation with strength values |
| `ChordsDescriptors` | Chord histogram, rate of change |
| `Dissonance` | Sensory roughness (0-1) from spectral peaks |
| `Inharmonicity` | Deviation from pure harmonics (0-1) |
| `PitchSalience` | How tonal/harmonic a sound is |
| `TuningFrequency` | Actual tuning frequency (non-440Hz detection) |
| `Tristimulus` | 3-element harmonic mixture vector |
| `OddToEvenHarmonicEnergyRatio` | Instrument identification |

#### Spectral Features

| Algorithm | Description |
|-----------|-------------|
| `SpectralContrast` | Peak-valley differences per band (texture) |
| `SpectralComplexity` | Number of peaks in spectrum |
| `Flux` | Frame-to-frame spectral change |
| `HFC` | High Frequency Content (brightness) |
| `FlatnessDB` | Noisiness vs tonality |
| `Entropy` | Signal complexity |
| `BarkBands` | Energy in Bark scale bands |
| `ERBBands` | Equivalent Rectangular Bandwidth bands |
| `MelBands` | Energy in mel-frequency bands |

#### Temporal/Envelope Features

| Algorithm | Description |
|-----------|-------------|
| `LogAttackTime` | Log10 of attack duration |
| `StrongDecay` | Decay characterization |
| `FadeDetection` | Fade-in/fade-out positions |
| `EffectiveDuration` | Non-silent duration |
| `SilenceRate` | Percentage of silent frames |

#### Audio Quality

| Algorithm | Description |
|-----------|-------------|
| `SNR` | Signal-to-noise ratio (frame-wise) |
| `ClickDetector` | Impulsive noise detection |
| `HumDetector` | Low-frequency hum detection |
| `SaturationDetector` | Clipping/saturation regions |
| `GapsDetector` | Silence gap detection |

#### Melody/Pitch

| Algorithm | Description |
|-----------|-------------|
| `PitchMelodia` | Predominant melody extraction |
| `PitchYin` / `PitchYinFFT` | Monophonic pitch detection |
| `Vibrato` | Vibrato detection (frequency + extent) |
| `MultiPitchMelodia` | Multiple pitch contours |

#### Music Similarity

| Algorithm | Description |
|-----------|-------------|
| `ChromaCrossSimilarity` | Binary chroma similarity matrix |
| `CoverSongSimilarity` | Cover song similarity score |
| `Chromaprinter` | Chromaprint audio fingerprint |

---

## Essentia TensorFlow - ML Models

**Models**: [essentia.upf.edu/models/](https://essentia.upf.edu/models/)

### Currently Using
- `discogs-effnet` (embedding model, 200-dim)
- `genre_discogs400` (400 Discogs music styles)
- `mood_aggressive`, `mood_happy`, `mood_relaxed`, `mood_sad`
- `danceability`
- `voice_instrumental`

### Feature Extractors / Embedding Models (Not Using)

| Model | Architecture | Description |
|-------|--------------|-------------|
| **MAEST** | Transformer | Music Audio Efficient Spectrogram Transformer - 5s, 10s, 20s, 30s versions |
| **OpenL3** | CNN | Self-supervised embeddings, `env` and `music` domains, 512/6144 dim |
| **VGGish** | VGG-based | AudioSet-trained embedding model |
| **MSD-MusiCNN** | MusiCNN | Million Song Dataset embeddings |

### Genre / Style Classification (Not Using)

| Model | Classes | Description |
|-------|---------|-------------|
| **genre_discogs519** | 519 | Extended Discogs taxonomy |
| **mtg_jamendo_genre** | 87 | MTG Jamendo genres |
| **genre_electronic** | ~? | Electronic music subgenres |
| **genre_dortmund** | ~10 | Dortmund dataset genres |
| **genre_tzanetakis** | 10 | Classic GTZAN genres |
| **fma_small** | ~8 | Free Music Archive genres |

### Mood / Emotion Models (Not Using)

| Model | Description |
|-------|-------------|
| **mood_acoustic** | Acoustic mood quality |
| **mood_electronic** | Electronic mood quality |
| **mood_party** | Party/upbeat mood detection |
| **moods_mirex** | 5 MIREX mood clusters |
| **emomusic** | Arousal/valence prediction |
| **deam** | Database for Emotional Analysis - arousal/valence |

### Music Perception / Quality (Not Using)

| Model | Description |
|-------|-------------|
| **approachability** | Mainstream vs. niche appeal (2-class, 3-class, regression) |
| **engagement** | Active vs. background listening suitability |
| **tonal_atonal** | Tonal vs. atonal classification |
| **timbre** | Timbre characteristics |

### Instrument Detection (Not Using)

| Model | Classes | Description |
|-------|---------|-------------|
| **mtg_jamendo_instrument** | 40 | Multi-label (guitar, piano, drums, synth, strings, etc.) |
| **nsynth_instrument** | ~11 | NSynth instrument families |
| **nsynth_acoustic_electronic** | 2 | Acoustic vs. electronic sound |
| **nsynth_bright_dark** | 2 | Bright vs. dark timbre |
| **nsynth_reverb** | 2 | Reverb presence detection |

### Audio Tagging / Multi-Label (Not Using)

| Model | Tags | Description |
|-------|------|-------------|
| **mtg_jamendo_moodtheme** | ~56 | Mood/theme multi-label tagging |
| **mtg_jamendo_top50tags** | 50 | Top 50 music tags from Jamendo |
| **msd** | 50 | Million Song Dataset tags |
| **mtt** | 50 | MagnaTagATune tags |

### Audio Event Recognition (Not Using)

| Model | Classes | Description |
|-------|---------|-------------|
| **YAMNet** | 521 | AudioSet audio event classes |
| **FSD-SINet** | ~200 | FreeSound environmental sounds |
| **urbansound8k** | 10 | Urban sounds |

### Specialized Models (Not Using)

| Model | Description |
|-------|-------------|
| **CREPE** | Pitch/F0 detection (5 sizes: tiny, small, medium, large, full) |
| **TempoCNN** | CNN-based tempo estimation |
| **gender** | Vocal gender classification |

---

## Madmom - Beat/Music Analysis

**Repository**: [github.com/CPJKU/madmom](https://github.com/CPJKU/madmom)
**Documentation**: [madmom.readthedocs.io](https://madmom.readthedocs.io/)

### Currently Using
Indirectly via allin1 for beat/downbeat tracking.

### Direct Use Value

#### Beat Tracking
| Processor | Description |
|-----------|-------------|
| `RNNBeatProcessor` | RNN-based beat activation (offline/online) |
| `DBNBeatTrackingProcessor` | Dynamic Bayesian Network beat tracking - **SOTA (MIREX 2015 winner)** |
| `BeatDetectionProcessor` | Beat detection with constant tempo |
| `CRFBeatDetectionProcessor` | Conditional Random Field beat tracking |

#### Downbeat/Bar Tracking
| Processor | Description |
|-----------|-------------|
| `RNNDownBeatProcessor` | Joint beat + downbeat activation |
| `DBNDownBeatTrackingProcessor` | DBN-based downbeat tracking |
| `DBNBarTrackingProcessor` | HMM-based bar structure tracking |

#### Tempo Estimation
| Processor | Description |
|-----------|-------------|
| `TempoEstimationProcessor` | Main tempo processor (comb/acf/dbn methods) |
| `ACFTempoHistogramProcessor` | Autocorrelation-based tempo histogram |

#### Chord Recognition
| Processor | Description |
|-----------|-------------|
| `DeepChromaChordRecognitionProcessor` | Deep chroma + CRF for major/minor chords |
| `CNNChordFeatureProcessor` | CNN-based chord feature extraction |
| `CRFChordRecognitionProcessor` | CRF chord recognition |

#### Key Detection
| Processor | Description |
|-----------|-------------|
| `CNNKeyRecognitionProcessor` | CNN-based global key recognition (24 keys) |

#### Note/Pitch Transcription
| Processor | Description |
|-----------|-------------|
| `RNNPianoNoteProcessor` | RNN-based piano note activation (88 keys) |
| `CNNPianoNoteProcessor` | CNN-based piano note activation |
| `ADSRNoteTrackingProcessor` | ADSR envelope-based HMM note tracking - **SOTA for piano** |

#### Onset Detection
| Processor | Description |
|-----------|-------------|
| `SpectralOnsetProcessor` | Spectral flux, superflux, phase-based |
| `RNNOnsetProcessor` | RNN-based onset activation |
| `CNNOnsetProcessor` | CNN-based onset detection - **MIREX 2016 winner** |

### Example: Chord Detection

```python
from madmom.features.chords import DeepChromaProcessor, DeepChromaChordRecognitionProcessor
from madmom.processors import SequentialProcessor

proc = SequentialProcessor([
    DeepChromaProcessor(),
    DeepChromaChordRecognitionProcessor()
])

chords = proc('track.mp3')
# Returns: [(start_time, end_time, chord_label), ...]
# e.g., [(0.0, 2.5, 'C:maj'), (2.5, 5.0, 'G:maj'), ...]
```

---

## Librosa - Audio Analysis

**Documentation**: [librosa.org/doc/](https://librosa.org/doc/)
**Repository**: [github.com/librosa/librosa](https://github.com/librosa/librosa)

### Currently Using (in fallback code)
- `beat_track` - tempo and beat detection
- `chroma_stft` - chroma features
- `mfcc` - mel-frequency cepstral coefficients
- `spectral_centroid`, `spectral_rolloff`
- `zero_crossing_rate`
- `rms` - RMS energy

### High Value Additions

#### Tonal Features

| Function | Description |
|----------|-------------|
| `tonnetz` | 6D tonal space (fifths, minor thirds, major thirds) - excellent for harmonic similarity |
| `chroma_cens` | Chroma Energy Normalized - robust for cover detection |
| `chroma_cqt` | Constant-Q chromagram (better low-frequency resolution) |

#### Spectral Features

| Function | Description |
|----------|-------------|
| `spectral_contrast` | 7-band peak-valley texture |
| `spectral_flatness` | Tonal vs noise-like (0=tonal, 1=noise) |
| `spectral_bandwidth` | Spectral spread |
| `melspectrogram` | Mel-scaled spectrogram |

#### Rhythm Features

| Function | Description |
|----------|-------------|
| `tempogram` | Tempo variation over time |
| `fourier_tempogram` | Short-time Fourier rhythm analysis |
| `plp` | Predominant Local Pulse |

#### Beat/Onset Detection

| Function | Description |
|----------|-------------|
| `onset_detect` | Locate note onset events |
| `onset_strength` | Spectral flux onset envelope |

#### Pitch Tracking

| Function | Description |
|----------|-------------|
| `pyin` | Probabilistic YIN - robust F0 estimation |
| `yin` | YIN fundamental frequency |
| `estimate_tuning` | Tuning deviation from A440 |

#### Effects/Decomposition

| Function | Description |
|----------|-------------|
| `hpss` | Harmonic-percussive source separation (fast, lightweight) |
| `harmonic` / `percussive` | Extract components |
| `trim` | Trim silence |

#### Structure Analysis

| Function | Description |
|----------|-------------|
| `recurrence_matrix` | Self-similarity matrix |
| `agglomerative` | Bottom-up segmentation |

#### Feature Engineering

| Function | Description |
|----------|-------------|
| `delta` | Compute derivatives (temporal dynamics) |
| `stack_memory` | Stack short-term history (context) |

### Example: Extended Analysis

```python
import librosa
import numpy as np

y, sr = librosa.load('track.mp3', sr=44100)

# Tonal
tonnetz = librosa.feature.tonnetz(y=y, sr=sr)  # 6 features

# Spectral
spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)  # 7 bands
spectral_flatness = librosa.feature.spectral_flatness(y=y)

# Rhythm
tempogram = librosa.feature.tempogram(y=y, sr=sr)
onset_env = librosa.onset.onset_strength(y=y, sr=sr)
onsets = librosa.onset.onset_detect(y=y, sr=sr)

# Pitch
f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=50, fmax=2000)

# Harmonic/Percussive ratio
y_harmonic, y_percussive = librosa.effects.hpss(y)
harmonic_ratio = np.mean(np.abs(y_harmonic)) / np.mean(np.abs(y))

# Tuning
tuning = librosa.estimate_tuning(y=y, sr=sr)  # cents from A440
```

---

## Feature Priority Recommendations

### Tier 1 - Essential (DJ-focused)

| Feature | Library | Description |
|---------|---------|-------------|
| Chord detection | madmom | `DeepChromaChordRecognitionProcessor` |
| Instrument detection | essentia-tf | `mtg_jamendo_instrument` (40 instruments) |
| LUFS loudness | essentia | `LoudnessEBUR128` |
| Electronic subgenres | essentia-tf | `genre_electronic` |
| Party/energy mood | essentia-tf | `mood_party` |
| Spectral contrast | essentia/librosa | Texture descriptor |
| Onset rate | essentia | Rhythmic density |

### Tier 2 - Enhanced Analysis

| Feature | Library | Description |
|---------|---------|-------------|
| MAEST embeddings | essentia-tf | Transformer embeddings (better than effnet) |
| Tonnetz | librosa | 6D harmonic space |
| Tempogram | librosa | Tempo variation over time |
| Approachability | essentia-tf | Mainstream vs underground |
| Engagement | essentia-tf | Active vs background listening |
| Attack time | essentia | `LogAttackTime` |
| Fade detection | essentia | Intro/outro detection |

### Tier 3 - Quality/Polish

| Feature | Library | Description |
|---------|---------|-------------|
| True peak detection | essentia | `TruePeakDetector` |
| Tuning detection | essentia/librosa | Non-440Hz detection |
| SNR | essentia | Audio quality metric |
| 6-stem separation | demucs | Guitar + piano stems |
| allin1 embeddings | allin1 | 24-dim per stem |
| BS-RoFormer vocals | audio-separator | Higher quality vocal isolation |

### Tier 4 - Specialized

| Feature | Library | Description |
|---------|---------|-------------|
| Piano transcription | madmom | Note-level MIDI output |
| Pitch tracking (pYIN) | librosa | Vocal melody extraction |
| Cover detection | essentia | `CoverSongSimilarity` |
| Audio fingerprint | essentia | `Chromaprinter` |
| Audio events | essentia-tf | `YAMNet` (521 classes) |

---

## Package Design Notes

### Pydantic Config Structure

The package should use a Pydantic config that allows turning features on/off:

```python
from pydantic import BaseModel
from typing import Optional

class AnalysisConfig(BaseModel):
    # Structure
    extract_beats: bool = True
    extract_segments: bool = True
    include_allin1_embeddings: bool = False

    # Stems
    extract_stems: bool = True
    stem_model: str = "htdemucs"  # or "htdemucs_ft", "htdemucs_6s"
    high_quality_vocals: bool = False  # Use BS-RoFormer

    # Tonal
    extract_key: bool = True
    extract_chords: bool = True
    extract_tuning: bool = False

    # Rhythm
    extract_tempo: bool = True
    extract_onset_rate: bool = True
    extract_tempogram: bool = False

    # Spectral
    extract_spectral_features: bool = True
    extract_spectral_contrast: bool = True

    # Loudness
    extract_loudness_ebur128: bool = True
    extract_dynamic_complexity: bool = True

    # ML Classification
    extract_genre: bool = True
    genre_model: str = "discogs400"  # or "discogs519", "electronic"
    extract_mood: bool = True
    extract_instruments: bool = True
    extract_danceability: bool = True
    extract_voice_instrumental: bool = True
    extract_approachability: bool = False
    extract_engagement: bool = False

    # Embeddings
    embedding_model: str = "discogs-effnet"  # or "maest", "openl3"

    # Quality
    extract_audio_quality: bool = False  # SNR, clicks, clipping

    # Advanced
    extract_pitch_melody: bool = False
    extract_piano_transcription: bool = False
```

### Modal Architecture

Two-stage GPU pipeline (current architecture is good):

1. **Stage 1 - PyTorch (A10G, CUDA 12.4)**
   - allin1 (structure)
   - demucs (stems)
   - audio-separator/BS-RoFormer (optional high-quality vocals)
   - madmom (chords, if enabled)
   - essentia traditional (DSP features)

2. **Stage 2 - TensorFlow (T4, CUDA 11.8)**
   - essentia-tensorflow models
   - All ML classification
   - Embeddings

### Output Schema

```python
class AnalysisResult(BaseModel):
    # Metadata
    filename: str
    duration: float
    sample_rate: int

    # Structure
    bpm: Optional[float]
    beats: Optional[list[float]]
    downbeats: Optional[list[float]]
    segments: Optional[list[Segment]]
    time_signature: Optional[str]

    # Tonal
    key: Optional[str]
    scale: Optional[str]
    key_confidence: Optional[float]
    chords: Optional[list[Chord]]
    tuning_deviation: Optional[float]

    # Rhythm
    onset_rate: Optional[float]
    tempo_stability: Optional[float]

    # Spectral
    spectral_centroid_mean: Optional[float]
    spectral_contrast: Optional[list[float]]
    spectral_flatness: Optional[float]

    # Loudness
    loudness_integrated: Optional[float]  # LUFS
    loudness_range: Optional[float]
    dynamic_complexity: Optional[float]
    true_peak: Optional[float]

    # ML Classification
    genres: Optional[list[GenrePrediction]]
    mood: Optional[MoodScores]
    instruments: Optional[list[InstrumentPrediction]]
    danceability: Optional[float]
    voice_instrumental: Optional[str]
    approachability: Optional[float]
    engagement: Optional[float]

    # Embeddings
    embedding: Optional[list[float]]
    embedding_model: Optional[str]

    # Stems (paths or bytes)
    stems: Optional[dict[str, str]]

    # Quality
    snr: Optional[float]
    has_clipping: Optional[bool]
```
