# Tweet

Built a GPU-accelerated audio analysis CLI with @modal_labs

Packages together:
- allin1 (BPM, beats, song structure)
- BS-RoFormer (SOTA vocal separation, 12.9 dB SDR)
- Essentia + TensorFlow (genre, mood, instruments, embeddings)

One command, ~$0.02/track. Run without installing:

```
uvx --from git+https://github.com/nsthorat/modal-audio-analysis modal-audio-analysis analyze song.mp3 -o ./output
```

---

# Example Output

```bash
$ uvx --from git+https://github.com/nsthorat/modal-audio-analysis modal-audio-analysis analyze song.mp3 -o ./output
```

```
Analyzing: song.mp3
Running Modal GPU pipeline...

Saved: ./output/analysis.json
Saved: ./output/embeddings.npy (shape: (179, 1280))
Saved: ./output/stems/vocals.mp3
Saved: ./output/stems/instrumental.mp3

Analysis Summary
 BPM       128.0
 Beats     432 beats (first at 0.46s)
 Segments  8 segments
 Key       A minor

Song Structure (128 BPM, 432 beats)
┏━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━┓
┃ Section ┃ Time         ┃ Duration ┃ Beats ┃
┡━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━┩
│ Intro   │ 0:00 → 0:16  │ 16s      │ 32    │
│ Verse   │ 0:16 → 0:48  │ 32s      │ 64    │
│ Chorus  │ 0:48 → 1:20  │ 32s      │ 64    │
│ Verse   │ 1:20 → 1:52  │ 32s      │ 64    │
│ Chorus  │ 1:52 → 2:24  │ 32s      │ 64    │
│ Bridge  │ 2:24 → 2:40  │ 16s      │ 32    │
│ Chorus  │ 2:40 → 3:12  │ 32s      │ 64    │
│ End     │ 3:12 → 3:30  │ 18s      │ 48    │
└─────────┴──────────────┴──────────┴───────┘

Top Genres
┏━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━┓
┃ Genre                ┃ Probability ┃
┡━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━┩
│ Electronic---Techno  │ 85.91%      │
│ Electronic---House   │ 26.50%      │
│ Electronic---Trance  │ 18.50%      │
└──────────────────────┴─────────────┘

     Mood                Other              Instruments
┏━━━━━━━━━━━━┳━━━━━━━━┓  ┏━━━━━━━━━━━━━━┓   ┏━━━━━━━━━━━━━━┳━━━━━━━┓
┃ Mood       ┃ Score  ┃  ┃ Danceability ┃   ┃ Instrument   ┃ Prob  ┃
┡━━━━━━━━━━━━╇━━━━━━━━┩  ┃ 88.00%       ┃   ┡━━━━━━━━━━━━━━╇━━━━━━━┩
│ Happy      │ 72.00% │  ┣━━━━━━━━━━━━━━┫   │ synthesizer  │ 92.0% │
│ Relaxed    │ 45.00% │  ┃ Type         ┃   │ drum machine │ 85.0% │
│ Aggressive │ 12.00% │  ┃ Instrumental ┃   │ bass         │ 67.0% │
│ Sad        │ 8.00%  │  ┗━━━━━━━━━━━━━━┛   └──────────────┴───────┘
└────────────┴────────┘

Total time: 45.2s (~$0.02)
```
