"""
Generate a jazz melody solo given a chord progression using
the trained encoder-decoder Music Transformer.

Usage:
    python generate_seq2seq.py \
        --weights output/seq2seq/results/best_loss_weights.pt \
        --chords "Cm7 | F7 | Bbj7 | Ebj7 | Am7b5 | D7 | Gm7 | Gm7" \
        --tempo 215 \
        --output generated_solo.mid
"""
import argparse
import re
import pickle
import sys
import numpy as np
import torch

sys.path.insert(0, 'src')

from model.music_transformer_seq2seq import MusicTransformerSeq2Seq
from utilities.remi_constants import REMI_VOCAB_SIZE, REMI_TOKEN_PAD
from utilities.device import get_device

# ---------------------------------------------------------------------------
# 1. Load vocabulary mappings
# ---------------------------------------------------------------------------

def load_vocab(pkl_path='pickles/remi_wstruct_vocab.pkl'):
    # Vocab class must be importable for pickle to resolve it
    import importlib
    build_vocab = importlib.import_module('build_vocab')
    # Register in __main__ so pickle can find it
    import __main__
    __main__.Vocab = build_vocab.Vocab
    with open(pkl_path, 'rb') as f:
        vocab = pickle.load(f)
    return vocab


# ---------------------------------------------------------------------------
# 2. Parse chord symbol → (tone, type, slash)
# ---------------------------------------------------------------------------

# Map user-facing chord quality names to REMI Chord-Type_ suffixes.
# Order matters: longer matches first to avoid prefix collisions.
CHORD_TYPE_MAP = [
    # half-diminished
    ('m7b5',  'm7b5'),
    # diminished
    ('dim7',  'o7'),  ('dim', 'o'),  ('o7', 'o7'),  ('o', 'o'),
    # augmented
    ('+j7',   '+j7'), ('+7',  '+7'), ('+',   '+'),
    # minor-major 7
    ('-j7',   '-j7'), ('mj7', '-j7'),
    # minor 7
    ('-7',    '-7'),  ('m7',  '-7'),  ('min7', '-7'),
    # minor 6
    ('-6',    '-6'),  ('m6',  '-6'),
    # minor
    ('-',     '-'),   ('m',   '-'),   ('min', '-'),
    # major 7
    ('j7',    'j7'),  ('maj7', 'j7'), ('M7', 'j7'),
    # dominant 7 (must come after longer matches)
    ('7',     '7'),
    # major 6
    ('6',     '6'),
    # sus
    ('sus7',  'sus7'), ('sus', 'sus'),
]


def parse_chord_symbol(symbol):
    """
    Parse a chord symbol like 'Cm7', 'Am7b5', 'Bbj7', 'F7/A' into
    (tone, chord_type, slash).

    Returns REMI-compatible values:
      tone  → e.g. 'C', 'Bb', 'F#'
      typ   → e.g. '-7', 'm7b5', 'j7', '7', ''
      slash → e.g. 'C', 'A' (defaults to tone if no slash)
    """
    symbol = symbol.strip()

    # Handle slash bass: 'Cm7/Eb'
    slash_bass = ''
    if '/' in symbol:
        parts = symbol.split('/')
        symbol = parts[0]
        slash_bass = parts[1]

    # Extract root tone (1 or 2 chars: letter + optional # or b)
    if len(symbol) >= 2 and symbol[1] in ('#', 'b'):
        tone = symbol[:2]
        remainder = symbol[2:]
    else:
        tone = symbol[:1]
        remainder = symbol[1:]

    # Match chord quality against known types
    chord_type = ''
    for pattern, remi_type in CHORD_TYPE_MAP:
        if remainder == pattern:
            chord_type = remi_type
            break

    if not slash_bass:
        slash_bass = tone

    return tone, chord_type, slash_bass


# ---------------------------------------------------------------------------
# 3. Encode chord progression string → REMI token IDs
# ---------------------------------------------------------------------------

def get_tempo_tokens(vocab, bpm):
    """Find the closest Tempo-Class and Tempo bin for a given BPM."""
    # Tempo class boundaries from training data
    bounds = vocab.tempo_cls_bounds  # e.g. [50, 80, 100, 140, 180, 310]
    tempo_cls = 0
    for i in range(len(bounds) - 1):
        if bounds[i] <= bpm < bounds[i + 1]:
            tempo_cls = i
            break
    else:
        tempo_cls = len(bounds) - 2  # last class

    # Find closest tempo value in vocab
    best_key = None
    best_diff = float('inf')
    for key in vocab.event2idx:
        if key.startswith('Tempo_'):
            val = float(key.split('_')[1])
            if abs(val - bpm) < best_diff:
                best_diff = abs(val - bpm)
                best_key = key
    
    return vocab.event2idx[f'Tempo-Class_{tempo_cls}'], vocab.event2idx[best_key]


def parse_chord_line(line):
    """
    Parse a chord line like:
      'A1: | |Cm7    |F7    |Bbj7    |Ebj7    |Am7b5    |D7    |Gm7    |Gm7    | |'
    
    Returns list of bars, each bar is a list of chord symbols.
    """
    # Remove section label (e.g., 'A1:')
    if ':' in line:
        line = line.split(':', 1)[1]

    # Split by '|', strip, and filter empty
    bars_raw = [b.strip() for b in line.split('|')]
    bars_raw = [b for b in bars_raw if b]

    bars = []
    for bar_str in bars_raw:
        # Split chords within a bar by whitespace
        chords = bar_str.split()
        chords = [c.strip() for c in chords if c.strip()]
        if chords:
            bars.append(chords)

    return bars


def encode_bars(chord_lines, vocab, bpm=215):
    """
    Convert chord progression lines into a list of per-bar token lists.
    
    Returns: list of lists, where each inner list is the REMI tokens for one bar.
    """
    tempo_cls_id, tempo_id = get_tempo_tokens(vocab, bpm)
    beat_positions = [0, 16, 32, 48]  # 4/4 time

    all_bars = []

    for line in chord_lines:
        bars = parse_chord_line(line)
        for bar_chords in bars:
            bar_ids = []
            bar_ids.append(vocab.event2idx['Bar'])

            n_chords = len(bar_chords)
            if n_chords == 1:
                chord_beats = {0: bar_chords[0]}
            elif n_chords == 2:
                chord_beats = {0: bar_chords[0], 2: bar_chords[1]}
            elif n_chords == 3:
                chord_beats = {0: bar_chords[0], 1: bar_chords[1], 2: bar_chords[2]}
            elif n_chords >= 4:
                chord_beats = {i: bar_chords[i] for i in range(4)}
            else:
                chord_beats = {}

            for beat_idx, pos in enumerate(beat_positions):
                bar_ids.append(vocab.event2idx[f'Position_{pos}/64'])
                bar_ids.append(tempo_cls_id)
                bar_ids.append(tempo_id)

                if beat_idx in chord_beats:
                    tone, typ, slash = parse_chord_symbol(chord_beats[beat_idx])
                    bar_ids.append(vocab.event2idx[f'Chord-Tone_{tone}'])
                    bar_ids.append(vocab.event2idx[f'Chord-Type_{typ}'])
                    bar_ids.append(vocab.event2idx[f'Chord-Slash_{slash}'])

            all_bars.append(bar_ids)

    return all_bars


def bars_to_encoder_ids(bar_token_lists, max_seq_len=512):
    """Flatten per-bar token lists into one padded encoder input."""
    ids = []
    for bar_ids in bar_token_lists:
        ids.extend(bar_ids)
    if len(ids) > max_seq_len:
        ids = ids[:max_seq_len]
    else:
        ids = ids + [REMI_TOKEN_PAD] * (max_seq_len - len(ids))
    return ids


def encode_chord_progression(chord_lines, vocab, bpm=215, max_seq_len=512):
    """Convenience wrapper: chord lines → flat padded encoder IDs."""
    bar_lists = encode_bars(chord_lines, vocab, bpm)
    return bars_to_encoder_ids(bar_lists, max_seq_len)


# ---------------------------------------------------------------------------
# 4. Autoregressive generation
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate(model, encoder_ids, max_len=512, temperature=1.0, top_k=10):
    """
    Autoregressively generate melody tokens given encoder input.
    """
    model.eval()
    device = get_device()

    # Encoder input: (1, S)
    src = torch.tensor([encoder_ids], dtype=torch.long).to(device)
    src_pad_mask = (src == REMI_TOKEN_PAD)

    # Start decoder with PAD (same as teacher-forcing shift)
    generated = [REMI_TOKEN_PAD]

    for _ in range(max_len - 1):
        tgt = torch.tensor([generated], dtype=torch.long).to(device)

        logits = model(src, tgt,
                       src_key_padding_mask=src_pad_mask,
                       tgt_key_padding_mask=None,
                       memory_key_padding_mask=src_pad_mask)

        # Take logits at the last position
        next_logits = logits[0, -1, :] / temperature

        # Top-k filtering
        if top_k > 0:
            values, indices = torch.topk(next_logits, top_k)
            mask = torch.full_like(next_logits, float('-inf'))
            mask.scatter_(0, indices, values)
            next_logits = mask

        probs = torch.softmax(next_logits, dim=-1)
        next_token = torch.multinomial(probs, 1).item()

        # Stop on PAD
        if next_token == REMI_TOKEN_PAD:
            break

        generated.append(next_token)

    return generated[1:]  # Remove the initial PAD start token


@torch.no_grad()
def generate_chunked(model, bar_token_lists, vocab, max_seq_len=512, chunk_bars=8,
                     context_len=64, temperature=1.0, top_k=10):
    """
    Generate melody for progressions longer than 8 bars using sliding-window chunking.

    1. Split bars into 8-bar chunks
    2. Generate melody for each chunk
    3. Prime the decoder with the tail of the previous chunk for continuity
    4. Stop each chunk when the target bar count is reached
    5. Concatenate all new tokens
    """
    bar_token_id = vocab.event2idx['Bar']
    n_bars = len(bar_token_lists)
    all_melody = []
    prev_tail = []  # decoder context from previous chunk

    for chunk_start in range(0, n_bars, chunk_bars):
        chunk_end = min(chunk_start + chunk_bars, n_bars)
        chunk_bars_list = bar_token_lists[chunk_start:chunk_end]
        target_bars = len(chunk_bars_list)

        # Encode this chunk's chords
        encoder_ids = bars_to_encoder_ids(chunk_bars_list, max_seq_len)

        model.eval()
        device = get_device()
        src = torch.tensor([encoder_ids], dtype=torch.long).to(device)
        src_pad_mask = (src == REMI_TOKEN_PAD)

        # Prime decoder: PAD + previous chunk's tail tokens
        if prev_tail:
            prime = [REMI_TOKEN_PAD] + prev_tail[-context_len:]
        else:
            prime = [REMI_TOKEN_PAD]

        generated = list(prime)
        n_prime = len(generated)
        bar_count = 0

        for _ in range(max_seq_len - n_prime):
            tgt = torch.tensor([generated], dtype=torch.long).to(device)

            logits = model(src, tgt,
                           src_key_padding_mask=src_pad_mask,
                           tgt_key_padding_mask=None,
                           memory_key_padding_mask=src_pad_mask)

            next_logits = logits[0, -1, :] / temperature

            if top_k > 0:
                values, indices = torch.topk(next_logits, top_k)
                mask = torch.full_like(next_logits, float('-inf'))
                mask.scatter_(0, indices, values)
                next_logits = mask

            probs = torch.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, 1).item()

            if next_token == REMI_TOKEN_PAD:
                break

            # Count Bar tokens to limit output to the target bar count
            if next_token == bar_token_id:
                bar_count += 1
                if bar_count > target_bars:
                    break

            generated.append(next_token)

        # Keep only freshly generated tokens (skip PAD + prime)
        new_tokens = generated[n_prime:]
        all_melody.extend(new_tokens)
        prev_tail = new_tokens

        print(f'  Chunk bars {chunk_start+1}-{chunk_end}: {len(new_tokens)} new tokens')

    return all_melody


# ---------------------------------------------------------------------------
# 5. Tokens → MIDI
# ---------------------------------------------------------------------------

def tokens_to_midi(token_ids, vocab, bpm=215, output_path='generated_solo.mid'):
    """
    Convert generated REMI token IDs to a MIDI file.
    """
    try:
        import pretty_midi
    except ImportError:
        print('Error: pretty_midi not installed. Run: pip install pretty_midi')
        return

    idx2event = vocab.idx2event
    midi = pretty_midi.PrettyMIDI(initial_tempo=bpm)
    instrument = pretty_midi.Instrument(program=0, name='Melody')

    beat_duration = 60.0 / bpm  # duration of one beat in seconds
    slot_duration = beat_duration / 16.0  # duration of one 1/64 slot

    current_bar = -1
    current_position = 0
    current_velocity = 80
    current_note = None
    current_duration = 0

    for tok_id in token_ids:
        if tok_id >= len(idx2event):
            continue
        event = idx2event[tok_id]

        if event == 'Bar':
            current_bar += 1
            current_position = 0

        elif event.startswith('Position_'):
            pos = int(event.split('_')[1].split('/')[0])
            current_position = pos

        elif event.startswith('Note-Velocity_'):
            vel = int(event.split('_')[1])
            current_velocity = min(vel * 4, 127)  # scale back up

        elif event.startswith('Note-On_'):
            current_note = int(event.split('_')[1])

        elif event.startswith('Note-Duration_'):
            dur_slots = int(event.split('_')[1].split('/')[0])
            if current_note is not None:
                # Calculate time
                bar_start = current_bar * 4 * beat_duration  # 4 beats per bar
                note_start = bar_start + current_position * slot_duration
                note_end = note_start + dur_slots * slot_duration

                note = pretty_midi.Note(
                    velocity=current_velocity,
                    pitch=current_note,
                    start=note_start,
                    end=note_end
                )
                instrument.notes.append(note)
                current_note = None

    midi.instruments.append(instrument)
    midi.write(output_path)
    print(f'MIDI file saved to: {output_path}')
    print(f'Total notes generated: {len(instrument.notes)}')

    return midi


# ---------------------------------------------------------------------------
# 6. CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description='Generate melody from chord progression')
    parser.add_argument('--weights', type=str,
                        default='output/seq2seq/results/best_loss_weights.pt',
                        help='Path to model weights')
    parser.add_argument('--chords', type=str, nargs='+',
                        help='Chord progression lines (one per section)')
    parser.add_argument('--chords_file', type=str, default=None,
                        help='Path to a text file with chord progression (one line per section)')
    parser.add_argument('--tempo', type=int, default=215, help='BPM (default: 215)')
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--top_k', type=int, default=10)
    parser.add_argument('--max_len', type=int, default=512, help='Max decoder length')
    parser.add_argument('--output', type=str, default='generated_solo.mid')
    parser.add_argument('--vocab', type=str, default='pickles/remi_wstruct_vocab.pkl')

    # Model architecture (must match training)
    parser.add_argument('--n_layers', type=int, default=6)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--dim_feedforward', type=int, default=1024)
    parser.add_argument('--max_sequence', type=int, default=512)

    parser.add_argument('--force_cpu', action='store_true')
    return parser.parse_args()


def main():
    from utilities.device import use_cuda
    args = parse_args()

    if args.force_cpu:
        use_cuda(False)

    # Load vocab
    vocab = load_vocab(args.vocab)
    print(f'Vocab size: {len(vocab.event2idx)}')

    # Parse chord progression
    if args.chords_file:
        with open(args.chords_file) as f:
            chord_lines = [l.strip() for l in f if l.strip()]
    elif args.chords:
        chord_lines = args.chords
    else:
        print('Error: provide --chords or --chords_file')
        return

    print(f'Chord lines: {len(chord_lines)}')
    for line in chord_lines:
        print(f'  {line}')

    # Encode chords into per-bar token lists
    bar_token_lists = encode_bars(chord_lines, vocab, bpm=args.tempo)
    n_bars = len(bar_token_lists)
    total_tokens = sum(len(b) for b in bar_token_lists)
    print(f'Total bars: {n_bars} ({total_tokens} encoder tokens)')

    # Load model
    model = MusicTransformerSeq2Seq(
        n_layers=args.n_layers, num_heads=args.num_heads,
        d_model=args.d_model, dim_feedforward=args.dim_feedforward,
        max_sequence=args.max_sequence
    ).to(get_device())

    state_dict = torch.load(args.weights, map_location=get_device())
    model.load_state_dict(state_dict)
    print(f'Model loaded from: {args.weights}')

    # Generate: single-shot for ≤ 8 bars, chunked for longer
    print(f'Generating melody (temp={args.temperature}, top_k={args.top_k})...')
    if n_bars <= 8:
        encoder_ids = bars_to_encoder_ids(bar_token_lists)
        melody_ids = generate(model, encoder_ids, max_len=args.max_len,
                              temperature=args.temperature, top_k=args.top_k)
    else:
        melody_ids = generate_chunked(model, bar_token_lists, vocab,
                                       temperature=args.temperature,
                                       top_k=args.top_k)

    print(f'Total generated: {len(melody_ids)} tokens')

    # Print generated tokens
    print('\nGenerated token preview (first 50):')
    for i, tok_id in enumerate(melody_ids[:50]):
        if tok_id < len(vocab.idx2event):
            print(f'  [{i:3d}] ID={tok_id:3d} -> {vocab.idx2event[tok_id]}')

    # Convert to MIDI
    tokens_to_midi(melody_ids, vocab, bpm=args.tempo, output_path=args.output)


if __name__ == '__main__':
    main()
