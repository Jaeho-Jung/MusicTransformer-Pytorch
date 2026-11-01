# Music Transformer — Jazz Solo Generation

PyTorch >= 1.2.0 · Python >= 3.6

This project builds on the [Music Transformer (Huang et al., 2018)](https://arxiv.org/abs/1809.04281) to generate jazz melody solos conditioned on chord progressions. An encoder-decoder architecture takes a chord progression as input and autoregressively generates a matching melody, trained on the [Weimar Jazz Database](https://jazzomat.hfm-weimar.de/dbformat/dboverview.html) using a Modified REMI token representation inspired by [BebopNet (Hakimi et al., 2020)](https://archives.ismir.net/ismir2020/paper/000132.pdf).

Two model variants are provided for comparison:

| Model | Attention | Description |
|-------|-----------|-------------|
| `MusicTransformerSeq2Seq` | RPR | Encoder-decoder with Relative Position Representation |
| `MusicTransformerBaseline` | Standard MHA | Same architecture, without RPR (ablation baseline) |

The training scripts expose matching default hyperparameters and use the same dataset class. The attention implementation differs; this repository alone does not establish that all reported runs used identical settings or random seeds.

---

## Table of Contents

- [Architecture](#architecture)
- [Modified REMI Representation](#modified-remi-representation)
- [Dataset: Weimar Jazz Database](#dataset-weimar-jazz-database)
- [Data Pipeline](#data-pipeline)
- [How to Run](#how-to-run)
- [Results](#results)
- [References](#references)

---

## Architecture

Both models use a full encoder-decoder structure:

```
Chord tokens  → Embedding → PositionalEncoding
    → Encoder  ─────────────────────────────┐
                                             ↓ cross-attention
Melody tokens → Embedding → PositionalEncoding
    → Decoder (causal mask) ←───────────────┘
    → Linear → Softmax → Next melody token
```

The encoder reads harmony tokens (chords, tempo) and the decoder autoregressively generates melody tokens conditioned on the encoder output.

**The only difference between the two models:**

| | `MusicTransformerSeq2Seq` | `MusicTransformerBaseline` |
|---|---|---|
| Encoder layers | `TransformerEncoderRPR` | `nn.TransformerEncoder` |
| Decoder layers | `TransformerDecoderRPR` | `nn.TransformerDecoder` |
| Positional info | Sinusoidal PE + learnable relative position embeddings | Sinusoidal PE only |

`MusicTransformerSeq2Seq` uses a custom `MultiheadAttentionRPR` (from `model/rpr.py`, adapted from PyTorch 1.2.0) that adds a learned relative position embedding matrix to the attention logits before softmax, so the model explicitly encodes the distance between any two tokens. `MusicTransformerBaseline` uses PyTorch's standard multi-head attention with no relative distance signal.

**Shared hyperparameters:**

| Parameter | Value |
|-----------|-------|
| `n_layers` | 6 |
| `num_heads` | 8 |
| `d_model` | 512 |
| `dim_feedforward` | 1024 |
| `dropout` | 0.1 |
| `max_sequence` | 512 |
| `batch_size` | 8 |
| `epochs` | 200 |
| `vocab_size` | 470 |

---

## Modified REMI Representation

The token representation is inspired by and based on [BebopNet (Hakimi et al., 2020)](https://archives.ismir.net/ismir2020/paper/000132.pdf), which introduced a REMI-style encoding for jazz improvisation that interleaves harmonic and melodic events. This project extends that scheme with additional structural and motivic tokens derived from the Weimar Jazz Database annotations.

### Token Categories

| Category | Tokens | Description |
|----------|:------:|-------------|
| `Note-On` | 128 | MIDI pitch 0–127 |
| `Note-Velocity` | 32 | Quantized velocity bins |
| `Note-Duration` | 32 | Duration in 1/64 beat units |
| `Position` | 64 | Timing offset within a bar (0/64–63/64) |
| `Bar` | 1 | Bar boundary marker |
| `Tempo-Class` | 6 | BPM range class (50, 80, 110, 140, 180, 320) |
| `Tempo` | 72 | Precise BPM bin (12 bins per class) |
| `Chord-Tone` | 12 | Root pitch class (C–B) |
| `Chord-Type` | ~23 | Chord quality (maj, min, 7, m7, j7, m7b5, dim, aug, sus, …) |
| `Chord-Slash` | 12 | Bass note for slash chords |
| `MLU-Type / Subtype` | — | Motivic unit type and subtype |
| `Backref-Distance` | 8 | How many bars back a motive refers to |
| `Backref-Variation` | — | Variation tag for motivic references |
| `Part-Start/End` | — | Form section boundaries (A, B, C) |
| `Rep-Start/End` | — | Repetition section markers |
| **Total** | **470** | |

### Encoder vs. Decoder Tokens

The encoder processes **harmony**: `Bar`, `Position`, `Tempo-Class`, `Tempo`, `Chord-Tone`, `Chord-Type`, `Chord-Slash`.

The decoder generates **melody**: `Bar`, `Position`, `Note-Velocity`, `Note-On`, `Note-Duration`, `MLU-*`, `Part-*`, `Rep-*`.

---

## Dataset: Weimar Jazz Database

Training data comes from the [Weimar Jazz Database (WJD)](https://jazzomat.hfm-weimar.de/dbformat/dboverview.html), which contains 456 annotated jazz solos by 78 performers.

Each piece is provided as two CSV files:
- `mcsv_beat/` — beat-level annotations: tempo, bar number, chord symbols, form labels
- `mcsv_melody/` — note-level annotations: pitch, onset, duration, loudness, motivic labels (MLU)

The CSV files are converted to interleaved REMI token sequences and saved as `seq2seq_data.npz` with separate `encoder_inputs` (chord) and `decoder_targets` (melody) arrays. Training uses 8-bar sliding windows with stride 1 and optional pitch augmentation (±3 semitones).

---

## Data Pipeline

```
mcsv_beat/ + mcsv_melody/   (456 pieces, Weimar Jazz Database)
        │
        ▼
  src/convert_to_remi.py
    ├─ collect_beats()     → tempo, bar markers
    ├─ collect_chords()    → Chord-Tone / -Type / -Slash tokens
    ├─ collect_notes()     → Note-On / -Velocity / -Duration tokens (1/64 grid)
    └─ collect_mlus()      → motivic structure tags
        │
        ▼
  src/build_vocab.py       → pickles/remi_wstruct_vocab.pkl  (470 tokens)
        │
        ▼
  seq2seq_data.npz
    encoder_inputs[N, 512]   ← harmony (chord) sequences
    decoder_targets[N, 512]  ← melody sequences
        │
        ▼
  dataset/seq2seq_dataset.py  (Seq2SeqDataset)
    · seeded 90/10 split of sequence pairs (not a solo-level split)
    · random pitch augmentation (±3 semitones) on Note-On / Chord-Tone / Chord-Slash
        │
        ▼
  train_seq2seq.py  /  train_baseline.py
```

---

## How to Run

### Build the REMI Dataset

The pre-built dataset is not tracked in git (large file). To regenerate it from the raw CSVs:
```bash
python src/convert_to_remi.py          # encodes all pieces to remi_encs_struct/
python src/build_seq2seq_dataset.py    # assembles seq2seq_data.npz
```

### Train

```bash
# Seq2seq with RPR (recommended)
python train_seq2seq.py \
    --data seq2seq_data/seq2seq_data.npz \
    --output_dir output/seq2seq \
    --augment

# Baseline (standard attention, no RPR)
python train_baseline.py \
    --data seq2seq_data/seq2seq_data.npz \
    --output_dir output/baseline \
    --augment
```

### Generate a Jazz Solo

Pass a chord progression using `|`-separated bar notation. Each bar can contain 1–4 chord symbols separated by spaces.

```bash
# F Blues (12 bars)
python generate_seq2seq.py \
    --weights output/seq2seq/results/best_loss_weights.pt \
    --chords "F7 | Bb7 | F7 | F7 | Bb7 | Bb7 | F7 | D7 | Gm7 | C7 | F7 C7 | Gm7 C7" \
    --tempo 180 \
    --output output/f_blues_solo.mid

# Autumn Leaves (8 bars)
python generate_seq2seq.py \
    --weights output/seq2seq/results/best_loss_weights.pt \
    --chords "Cm7 | F7 | Bbj7 | Ebj7 | Am7b5 | D7 | Gm7 | Gm7" \
    --tempo 215 \
    --output output/autumn_leaves_solo.mid

# From a chord file (one section per line, section labels optional)
python generate_seq2seq.py \
    --weights output/seq2seq/results/best_loss_weights.pt \
    --chords_file chords/there_will_never.txt \
    --tempo 230 \
    --output output/there_will_never_solo.mid
```

**Chord symbol syntax**

| Symbol | Meaning |
|--------|---------|
| `C`, `Cm`, `Cj7`, `C7` | Major, minor, major-7, dominant-7 |
| `Cm7`, `Cm7b5` | Minor-7, half-diminished |
| `Cdim`, `Co7` | Diminished, diminished-7 |
| `C+`, `C+7` | Augmented |
| `Csus`, `Csus7` | Suspended |
| `Cm7/Eb` | Slash chord (bass note after `/`) |
| `Bb`, `F#` | Flat / sharp accidentals |

For progressions longer than 8 bars, generation automatically uses a sliding-window approach with decoder context carry-over between chunks.

**Generation options**

| Flag | Default | Description |
|------|:-------:|-------------|
| `--temperature` | 1.0 | Sampling temperature |
| `--top_k` | 10 | Top-k sampling |
| `--max_len` | 512 | Maximum decoder token length |
| `--tempo` | 215 | BPM |

---

## Results

The table below records reported best validation losses. The training scripts allow up to 200 epochs and optional pitch augmentation (±3 semitones); the run logs and exact settings supporting these values are not included here. The dataset class splits sequence pairs at random, so overlapping windows from one solo may appear in both training and validation sets.

| Model | Best Val Loss | Epoch |
|-------|:------------:|:-----:|
| Baseline (standard MHA, no RPR) | 1.42 | 19 |
| Seq2Seq (RPR) | **1.03** | 14 |

In the reported runs, the RPR model had a lower best validation loss. This is not sufficient to establish a general improvement in harmony-conditioned melody generation. A solo-level split, saved run configurations, repeated seeds, and listening or music-structure evaluation are needed for a stronger conclusion.

---

## References

- Huang et al. (2018). [Music Transformer: Generating Music with Long-Term Structure](https://arxiv.org/abs/1809.04281).
- Shaw et al. (2018). [Self-Attention with Relative Position Representations](https://arxiv.org/abs/1803.02155).
- Hakimi, S. H., Bhonker, N., & El-Yaniv, R. (2020). [BebopNet: Deep Neural Models for Personalized Jazz Improvisations](https://archives.ismir.net/ismir2020/paper/000132.pdf). *Proceedings of ISMIR 2020*. *(Modified REMI representation)*
- Pfleiderer et al. (2017). [Inside the Jazzomat: New Perspectives for Jazz Research](https://schott-campus.com/jazzomat/). *(Weimar Jazz Database)*
