# convert_to_remi.py
# Converts annotated music CSV files (beat + melody) into REMI token encodings
# for use in transformer-based music generation models.
import pandas as pd
import numpy as np
import pickle, os
from glob import glob
from functools import cmp_to_key
import matplotlib.pyplot as plt

from chord_processor import ChordProcessor
from mlu_processor import MLUProcessor
from build_vocab import Vocab
from remi_containers import NoteREMI, ChordREMI, BeatREMI, sort_remi_events_cmp
from explore_mcsv import get_chords_info
from utils import clip_val, db2velocity


# ---------------------------------------------------------------------------
# Step 1: Collect beat events
# ---------------------------------------------------------------------------

def collect_beats(beat_df, vocab):
  """
  Build a dict of BeatREMI objects keyed by (bar, beat).

  REMI position quantization:
    - 1 bar = 64 slots, 1 beat = 16 slots
    - position = (beat_number - 1) * 16  →  0, 16, 32, 48

  Also classifies each beat's tempo into a class + bin via vocab bounds.
  """
  beat_dict = {}
  for i, beat in beat_df.iterrows():
    # BeatREMI(is_bar, bar, position, start_time, duration)
    # is_bar=True only for beat 1 (the downbeat)
    beat_dict[(int(beat['bar']), int(beat['beat']))] = \
      BeatREMI(beat['beat'] == 1, int(beat['bar']), (int(beat['beat']) - 1) * 16, beat['onset'], beat['duration'])
    
    # Classify tempo at this beat into a discrete class and bin
    beat_dict[(int(beat['bar']), int(beat['beat']))].get_tempo(vocab.tempo_cls_bounds, vocab.tempo_vals)

  return beat_dict


# ---------------------------------------------------------------------------
# Step 2: Collect structural segment tags (optional, used when use_structure=True)
# ---------------------------------------------------------------------------

def collect_segments(beat_df, beat_dict):
  """
  Patch form-segment labels (e.g. A, B, chorus) onto BeatREMI objects.

  Detects boundaries where the 'form' column changes and tags the adjacent
  beats with start_seg / end_seg so that Part-Start / Part-End tokens can
  be emitted later during encoding.
  """
  cur_seg = 'NA'

  for i, beat in beat_df.iterrows():
    # Detect a form change (but not at the very last row)
    if beat['form'] != cur_seg and i != len(beat_df) - 1:
      beat_key = (int(beat['bar']), int(beat['beat']))
      if cur_seg != 'NA':
        # Tag this beat as both the end of the previous segment and the start of the new one
        beat_dict[beat_key].patch_segment_tag(end_seg=cur_seg, start_seg=beat['form'])
      else:
        # First segment: only tag the start
        beat_dict[beat_key].patch_segment_tag(start_seg=beat['form'])
      cur_seg = beat['form']
    
    # Tag the very last beat as the end of the final segment
    if i == len(beat_df) - 1:
      beat_key = (int(beat['bar']), int(beat['beat']))
      beat_dict[beat_key].patch_segment_tag(end_seg=cur_seg)

  return beat_dict        


# ---------------------------------------------------------------------------
# Step 3: Collect chord events
# ---------------------------------------------------------------------------

def collect_chords(beat_df, chord_processor):
  """
  Parse chord annotations into ChordREMI objects.

  Chord literal format example: 'Bb6'  →  tone='Bb', typ='6', slash=''
  If no slash bass note is specified, it defaults to the root tone.

  REMI position: (beat - 1) * 16  (same 64-slot grid as beats)
  Skips 'NC' (No Chord) symbols.
  """
  # get_chords_info returns chord segments; repeat_long_chord=True fills long chords beat by beat
  chords = get_chords_info(beat_df, repeat_long_chord=True, repeat_beats=4)
  chords_remi = []

  for ch in chords:
    if ch.id == 'NC':
      # Skip "No Chord" regions
      continue
    else:
      # ex: Bb6 -> tone = 'Bb' typ = '6' slash = ''
      tone, typ, slash = chord_processor._parse_chord_literal(ch.id)

    if not slash:
      # ex: Bb6 -> tone = 'Bb' typ = '6' slash = 'Bb'
      slash = tone

    # ch.start_barbeat = (bar, beat); convert beat → REMI position slot
    # ChordREMI(tone, typ, slash, bar, position)
    chords_remi.append(
      ChordREMI(tone, typ, slash, ch.start_barbeat[0], (ch.start_barbeat[1] - 1) * 16)
    )

  return chords_remi


# ---------------------------------------------------------------------------
# Step 4: Collect note events
# ---------------------------------------------------------------------------

def collect_notes(melody_df, beat_dict, default_vel=20):
  """
  Convert raw melody note rows into NoteREMI objects.

  Quantization (both clamped to valid REMI ranges):
    - Duration : round(note_dur / beat_dur * 16), clamped to [1, 32] slots
    - Position : offset from beat onset → slots + beat's base position, clamped to [0, 63]

  Velocity:
    - Derived from 'loud_median' column (dB) via db2velocity, scaled to [0, 127//4]
    - Falls back to default_vel if the column is absent or NaN
  """
  notes = []
  n_short = 0  # Counter for extremely short notes (duration < 1ms), logged for debugging

  for i, note in melody_df.iterrows():
    if note['duration'] < 1e-3:
      n_short += 1  # Flag suspiciously short notes (likely annotation errors)

    # Look up the beat this note falls on
    beat = beat_dict[(int(note['bar']), int(note['beat']))]
    beat_dur = beat.duration

    # Quantize note duration into 1/16-beat slots, clamp to [1, 32]
    note_dur = round(note['duration'] / beat_dur * 16)
    note_dur = clip_val(note_dur, 1, 32)

    # Compute REMI position: fine offset within the beat + beat's base position in the bar
    note_pos = round((note['onset'] - beat.start_time) / beat.duration * 16) + beat.pos_remi.position
    note_pos = clip_val(note_pos, 0, 63)

    # Convert loudness (dB) to a coarse MIDI velocity bucket [0, 31]
    if 'loud_median' in melody_df.columns and not np.isnan(note['loud_median']):
      note_vel = db2velocity(abs(note['loud_median'])) // 4
    else:
      note_vel = default_vel

    notes.append(
      NoteREMI(int(note['pitch']), note_vel, int(note['bar']), note_pos, note_dur)
    )

  return notes, n_short


# ---------------------------------------------------------------------------
# Step 5: Attach MLU (Musical Linguistic Unit) tags (optional)
# ---------------------------------------------------------------------------

def collect_mlus(melody_df, notes, mlu_processor):
  """
  Annotate NoteREMI objects with motivic/phrase structure (MLU tags).

  MLU columns expected in melody_df:
    - 'idea'         : label for the current motivic unit (changes trigger a new tag)
    - 'phrase_begin' : 1 if this note starts a new phrase
    - 'phrase_id'    : (optional) phrase group identifier

  Back-references:
    - rep_backref points N positions back in mlu_pos to mark a motif as
      'referred-to', enabling the model to learn repetition/variation patterns.
  """
  # Gracefully handle missing columns
  if 'phrase_id' in melody_df.columns:
    assert 'phrase_begin' in melody_df.columns
  if 'phrase_begin' not in melody_df.columns:
    melody_df['phrase_begin'] = 0
  if 'idea' not in melody_df.columns:
    melody_df['idea'] = 'NA'

  cur_idea = 'NA'
  mlu_pos = []  # Tracks indices of notes that started a new MLU (for back-reference resolution)

  for i, note in melody_df.iterrows():
    # Sanity check: note order must match between melody_df and notes list
    assert notes[i].pitch == note['pitch']

    if note['idea'] != cur_idea:
      # A new motivic unit has started
      cur_idea = note['idea']
      is_phrase = bool(note['phrase_begin'])
      has_void, rep_backref, rep_variation, typ, sub_typ = mlu_processor.parse_mlu_literal(note['idea'])

      if rep_backref:
        if len(mlu_pos) < rep_backref:
          # Back-reference is out of range; clear it
          rep_backref, rep_variation = '', ''
        else:
          # Mark the referred-to note so the model knows it is being repeated/varied
          notes[ mlu_pos[-rep_backref] ].mlu_tag.referred = True

      notes[i].patch_mlu_tag(is_phrase, has_void, rep_backref, rep_variation, typ, sub_typ)
      mlu_pos.append(i)

  return notes


# ---------------------------------------------------------------------------
# Step 6: Convert REMI event objects into integer token rows (the actual encoding)
# ---------------------------------------------------------------------------

def event_to_encodings(events, vocab, chord_processor, use_structure=False):
  """
  Linearize a sorted list of REMI events into a DataFrame of token rows.

  Token schema by event type:
    beat  → [Part-End?, Part-Start?] [Bar?] [Position] [Tempo-Class] [Tempo]
    chord → [Position?] [Chord-Tone] [Chord-Type] [Chord-Slash]
    note  → [Position?] [Phrase?] [MLU tokens?] [Note-Velocity] [Note-On] [Note-Duration]

  Position deduplication:
    appeared_pos tracks which position slots have been emitted within the
    current bar; a Position token is only added the first time a new slot
    is encountered (avoids redundant Position tokens for simultaneous events).
  """
  rows = []
  appeared_pos = set()  # Reset at each new bar

  for i, ev in enumerate(events):

    # --- Beat event ---
    if ev.ev_type == 'beat':
      # Emit structural boundary tokens (Part/Rep) before the bar marker
      if use_structure and ev.segment_tag is not None and i != len(events) - 1:
        if ev.segment_tag.part_end:
          rows.append( {'EVENT': 'Rep-End',   'VALUE': ev.segment_tag.rep_end,   'VALUE_IDX': ev.segment_tag.rep_end,   'ENCODING': vocab.event2idx['Rep-End_{}'.format(ev.segment_tag.rep_end)]} )
          rows.append( {'EVENT': 'Part-End',  'VALUE': ev.segment_tag.part_end,  'VALUE_IDX': ev.segment_tag.part_end,  'ENCODING': vocab.event2idx['Part-End_{}'.format(ev.segment_tag.part_end)]} )
        if ev.segment_tag.part_start:
          rows.append( {'EVENT': 'Part-Start','VALUE': ev.segment_tag.part_start,'VALUE_IDX': ev.segment_tag.part_start,'ENCODING': vocab.event2idx['Part-Start_{}'.format(ev.segment_tag.part_start)]} )
          rows.append( {'EVENT': 'Rep-Start', 'VALUE': ev.segment_tag.rep_start, 'VALUE_IDX': ev.segment_tag.rep_start, 'ENCODING': vocab.event2idx['Rep-Start_{}'.format(ev.segment_tag.rep_start)]} )

      if ev.is_bar:
        # Reset position tracker at the start of every new bar
        appeared_pos = set()
        rows.append( {'EVENT': 'Bar', 'VALUE': 0, 'VALUE_IDX': 0, 'ENCODING': vocab.event2idx['Bar']} )

      appeared_pos.add( ev.pos_remi.position )
      rows.append( {'EVENT': 'Position',    'VALUE': ev.pos_remi.position, 'VALUE_IDX': ev.pos_remi.position, 'ENCODING': vocab.event2idx['Position_{}/64'.format(ev.pos_remi.position)]})
      rows.append( {'EVENT': 'Tempo-Class', 'VALUE': ev.tempo_cls,         'VALUE_IDX': ev.tempo_cls,         'ENCODING': vocab.ev_type_base['Tempo-Class'] + ev.tempo_cls} )
      rows.append( {'EVENT': 'Tempo',       'VALUE': vocab.idx2event[ vocab.ev_type_base['Tempo'] + ev.tempo_bin ].split('_')[-1], 'VALUE_IDX': ev.tempo_bin, 'ENCODING': vocab.ev_type_base['Tempo'] + ev.tempo_bin} )

      # Emit closing structural tokens at the very last event
      if use_structure and ev.segment_tag is not None and i == len(events) - 1:
        rows.append( {'EVENT': 'Rep-End',  'VALUE': ev.segment_tag.rep_end,  'VALUE_IDX': ev.segment_tag.rep_end,  'ENCODING': vocab.event2idx['Rep-End_{}'.format(ev.segment_tag.rep_end)]} )
        rows.append( {'EVENT': 'Part-End', 'VALUE': ev.segment_tag.part_end, 'VALUE_IDX': ev.segment_tag.part_end, 'ENCODING': vocab.event2idx['Part-End_{}'.format(ev.segment_tag.part_end)]} )

    # --- Chord event ---
    elif ev.ev_type == 'chord':
      # Only emit Position if this slot hasn't been announced yet in this bar
      if ev.pos_remi.position not in appeared_pos:
        appeared_pos.add( ev.pos_remi.position )
        rows.append( {'EVENT': 'Position', 'VALUE': ev.pos_remi.position, 'VALUE_IDX': ev.pos_remi.position, 'ENCODING': vocab.event2idx['Position_{}/64'.format(ev.pos_remi.position)]})
      
      tone_idx, slash_idx = chord_processor.key_map[ev.tone], chord_processor.key_map[ev.slash]
      rows.append( {'EVENT': 'Chord-Tone',  'VALUE': ev.tone,  'VALUE_IDX': tone_idx,  'ENCODING': vocab.event2idx['Chord-Tone_{}'.format(ev.tone)]})
      rows.append( {'EVENT': 'Chord-Type',  'VALUE': ev.typ,   'VALUE_IDX': 'None',    'ENCODING': vocab.event2idx['Chord-Type_{}'.format(ev.typ)]} )
      rows.append( {'EVENT': 'Chord-Slash', 'VALUE': ev.slash, 'VALUE_IDX': slash_idx, 'ENCODING': vocab.event2idx['Chord-Slash_{}'.format(ev.slash)]} )
      
    # --- Note event ---
    elif ev.ev_type == 'note':
      # Only emit Position if this slot hasn't been announced yet in this bar
      if ev.pos_remi.position not in appeared_pos:
        appeared_pos.add( ev.pos_remi.position )
        rows.append( {'EVENT': 'Position', 'VALUE': ev.pos_remi.position, 'VALUE_IDX': ev.pos_remi.position, 'ENCODING': vocab.event2idx['Position_{}/64'.format(ev.pos_remi.position)]})
      
      # Optional structural / motivic tokens
      if use_structure and ev.mlu_tag is not None:
        if ev.mlu_tag.is_phrase:
          rows.append( {'EVENT': 'Phrase', 'VALUE': 0, 'VALUE_IDX': 0, 'ENCODING': vocab.event2idx['Phrase']} )
        
        rows.append( {'EVENT': 'MLU-Type', 'VALUE': ev.mlu_tag.typ, 'VALUE_IDX': 0, 'ENCODING': vocab.event2idx['MLU-Type_{}'.format(ev.mlu_tag.typ)]} )

        # Emit sub-type if present; fall back to 'general' if the vocab has it
        if ev.mlu_tag.sub_typ:
          rows.append( {'EVENT': 'MLU-Subtype', 'VALUE': ev.mlu_tag.sub_typ, 'VALUE_IDX': 0, 'ENCODING': vocab.event2idx['MLU-Subtype_{}-{}'.format(ev.mlu_tag.typ, ev.mlu_tag.sub_typ)]} )
        elif not ev.mlu_tag.sub_typ and 'MLU-Subtype_{}-general'.format(ev.mlu_tag.typ) in vocab.event2idx.keys():
          rows.append( {'EVENT': 'MLU-Subtype', 'VALUE': 'general', 'VALUE_IDX': 0, 'ENCODING': vocab.event2idx['MLU-Subtype_{}-general'.format(ev.mlu_tag.typ)]} )

        # Repetition / variation back-reference tokens
        if ev.mlu_tag.rep_backref:
          rows.append( {'EVENT': 'Backref-Distance',  'VALUE': ev.mlu_tag.rep_backref,  'VALUE_IDX': int(ev.mlu_tag.rep_backref),  'ENCODING': vocab.event2idx['Backref-Distance_{}'.format(ev.mlu_tag.rep_backref)]} )
        if ev.mlu_tag.rep_variation:
          rows.append( {'EVENT': 'Backref-Variation', 'VALUE': ev.mlu_tag.rep_variation, 'VALUE_IDX': 0, 'ENCODING': vocab.event2idx['Backref-Variation_{}'.format(ev.mlu_tag.rep_variation)]} )
        
        # Mark this note as previously referenced by another MLU
        if ev.mlu_tag.referred:
          rows.append( {'EVENT': 'Backref-Referred', 'VALUE': 0, 'VALUE_IDX': 0, 'ENCODING': vocab.event2idx['Backref-Referred']} )

      # Core note tokens
      rows.append( {'EVENT': 'Note-Velocity', 'VALUE': ev.velocity, 'VALUE_IDX': ev.velocity, 'ENCODING': vocab.event2idx['Note-Velocity_{}'.format(ev.velocity)]})
      rows.append( {'EVENT': 'Note-On',       'VALUE': ev.pitch,    'VALUE_IDX': ev.pitch,    'ENCODING': vocab.event2idx['Note-On_{}'.format(ev.pitch)]} )
      rows.append( {'EVENT': 'Note-Duration', 'VALUE': ev.duration, 'VALUE_IDX': ev.duration, 'ENCODING': vocab.event2idx['Note-Duration_{}/64'.format(ev.duration)]} )

  df_enc = pd.DataFrame(rows, columns=['EVENT', 'VALUE', 'VALUE_IDX', 'ENCODING'])
  print (df_enc.info())

  return df_enc


# ---------------------------------------------------------------------------
# Step 7: Seq2Seq data pipeline (encoder=harmony, decoder=melody)
# ---------------------------------------------------------------------------

def group_events_by_bar(beats, chords, notes):
  """
  Group collected REMI events by bar number.

  Returns:
    dict[bar] -> {'beats': [...], 'chords': [...], 'notes': [...]}
    Each sub-list is sorted by position within the bar.
  """
  events_by_bar = {}

  for b in beats:
    bar = b.pos_remi.bar
    if bar not in events_by_bar:
      events_by_bar[bar] = {'beats': [], 'chords': [], 'notes': []}
    events_by_bar[bar]['beats'].append(b)

  for c in chords:
    bar = c.pos_remi.bar
    if bar in events_by_bar:
      events_by_bar[bar]['chords'].append(c)

  for n in notes:
    bar = n.pos_remi.bar
    if bar in events_by_bar:
      events_by_bar[bar]['notes'].append(n)

  # Sort sub-lists by position
  for bar in events_by_bar:
    events_by_bar[bar]['beats'].sort(key=lambda x: x.pos_remi.position)
    events_by_bar[bar]['chords'].sort(key=lambda x: x.pos_remi.position)
    events_by_bar[bar]['notes'].sort(key=lambda x: (x.pos_remi.position, x.pitch))

  return events_by_bar


def encode_encoder_events(events_by_bar, window_bars, vocab, chord_processor, use_structure=False):
  """
  Encode harmony events for the encoder: Bar, Position, Tempo, Chords, Segments.

  Token order per beat position:
    [Part-End?] [Part-Start?] [Rep-End?] [Rep-Start?]
    [Bar?] [Position] [Tempo-Class] [Tempo]
    [Chord-Tone] [Chord-Type] [Chord-Slash]
  """
  ids = []
  appeared_pos = set()

  for bar in window_bars:
    if bar not in events_by_bar:
      continue

    bar_beats  = events_by_bar[bar]['beats']
    bar_chords = events_by_bar[bar]['chords']

    # Merge beats and chords, sorted by position (beats first at same pos)
    beat_idx, chord_idx = 0, 0
    merged = []
    while beat_idx < len(bar_beats) and chord_idx < len(bar_chords):
      if bar_beats[beat_idx].pos_remi.position <= bar_chords[chord_idx].pos_remi.position:
        merged.append(bar_beats[beat_idx])
        beat_idx += 1
      else:
        merged.append(bar_chords[chord_idx])
        chord_idx += 1
    merged.extend(bar_beats[beat_idx:])
    merged.extend(bar_chords[chord_idx:])

    for ev in merged:
      if ev.ev_type == 'beat':
        # Structural boundary tokens
        if use_structure and ev.segment_tag is not None:
          if ev.segment_tag.part_end:
            ids.append( vocab.event2idx['Rep-End_{}'.format(ev.segment_tag.rep_end)] )
            ids.append( vocab.event2idx['Part-End_{}'.format(ev.segment_tag.part_end)] )
          if ev.segment_tag.part_start:
            ids.append( vocab.event2idx['Part-Start_{}'.format(ev.segment_tag.part_start)] )
            ids.append( vocab.event2idx['Rep-Start_{}'.format(ev.segment_tag.rep_start)] )

        if ev.is_bar:
          appeared_pos = set()
          ids.append( vocab.event2idx['Bar'] )

        appeared_pos.add( ev.pos_remi.position )
        ids.append( vocab.event2idx['Position_{}/64'.format(ev.pos_remi.position)] )
        ids.append( vocab.ev_type_base['Tempo-Class'] + ev.tempo_cls )
        ids.append( vocab.ev_type_base['Tempo'] + ev.tempo_bin )

      elif ev.ev_type == 'chord':
        if ev.pos_remi.position not in appeared_pos:
          appeared_pos.add( ev.pos_remi.position )
          ids.append( vocab.event2idx['Position_{}/64'.format(ev.pos_remi.position)] )

        ids.append( vocab.event2idx['Chord-Tone_{}'.format(ev.tone)] )
        ids.append( vocab.event2idx['Chord-Type_{}'.format(ev.typ)] )
        ids.append( vocab.event2idx['Chord-Slash_{}'.format(ev.slash)] )

  return ids


def encode_decoder_events(events_by_bar, window_bars, vocab, use_structure=False):
  """
  Encode melody events for the decoder: Bar, Position, Notes, MLUs.

  Token order per note position:
    [Bar?] [Position?]
    [Phrase?] [MLU-Type] [MLU-Subtype?] [Backref-*?]
    [Note-Velocity] [Note-On] [Note-Duration]
  """
  ids = []
  appeared_pos = set()

  for bar in window_bars:
    if bar not in events_by_bar:
      continue

    bar_beats = events_by_bar[bar]['beats']
    bar_notes = events_by_bar[bar]['notes']

    # Merge beats and notes, sorted by position (beats first at same pos)
    beat_idx, note_idx = 0, 0
    merged = []
    while beat_idx < len(bar_beats) and note_idx < len(bar_notes):
      if bar_beats[beat_idx].pos_remi.position <= bar_notes[note_idx].pos_remi.position:
        merged.append(bar_beats[beat_idx])
        beat_idx += 1
      else:
        merged.append(bar_notes[note_idx])
        note_idx += 1
    merged.extend(bar_beats[beat_idx:])
    merged.extend(bar_notes[note_idx:])

    for ev in merged:
      if ev.ev_type == 'beat':
        if ev.is_bar:
          appeared_pos = set()
          ids.append( vocab.event2idx['Bar'] )

        appeared_pos.add( ev.pos_remi.position )
        ids.append( vocab.event2idx['Position_{}/64'.format(ev.pos_remi.position)] )

      elif ev.ev_type == 'note':
        if ev.pos_remi.position not in appeared_pos:
          appeared_pos.add( ev.pos_remi.position )
          ids.append( vocab.event2idx['Position_{}/64'.format(ev.pos_remi.position)] )

        # MLU tokens
        if use_structure and ev.mlu_tag is not None:
          if ev.mlu_tag.is_phrase:
            ids.append( vocab.event2idx['Phrase'] )

          ids.append( vocab.event2idx['MLU-Type_{}'.format(ev.mlu_tag.typ)] )

          if ev.mlu_tag.sub_typ:
            ids.append( vocab.event2idx['MLU-Subtype_{}-{}'.format(ev.mlu_tag.typ, ev.mlu_tag.sub_typ)] )
          elif not ev.mlu_tag.sub_typ and 'MLU-Subtype_{}-general'.format(ev.mlu_tag.typ) in vocab.event2idx:
            ids.append( vocab.event2idx['MLU-Subtype_{}-general'.format(ev.mlu_tag.typ)] )

          if ev.mlu_tag.rep_backref:
            ids.append( vocab.event2idx['Backref-Distance_{}'.format(ev.mlu_tag.rep_backref)] )
          if ev.mlu_tag.rep_variation:
            ids.append( vocab.event2idx['Backref-Variation_{}'.format(ev.mlu_tag.rep_variation)] )
          if ev.mlu_tag.referred:
            ids.append( vocab.event2idx['Backref-Referred'] )

        # Note tokens
        ids.append( vocab.event2idx['Note-Velocity_{}'.format(ev.velocity)] )
        ids.append( vocab.event2idx['Note-On_{}'.format(ev.pitch)] )
        ids.append( vocab.event2idx['Note-Duration_{}/64'.format(ev.duration)] )

  return ids


def pad_or_truncate(ids, max_seq_len, pad_id):
  """Pad with pad_id or truncate to exactly max_seq_len."""
  if len(ids) >= max_seq_len:
    return ids[:max_seq_len]
  else:
    return ids + [pad_id] * (max_seq_len - len(ids))


def convert_piece_seq2seq(beat_csv, melody_csv, vocab, chord_processor,
                          window_bars=8, stride=1, max_seq_len=512,
                          accept_timesig=["'4/4'"],
                          use_structure=False, mlu_processor=None):
  """
  Convert a single piece into seq2seq training pairs.

  Returns a list of (encoder_ids, decoder_ids) tuples, each of length max_seq_len.
  Returns empty list if the piece is skipped (wrong time signature, too short).
  """
  beat_df   = pd.read_csv(beat_csv,   encoding='utf-8')
  melody_df = pd.read_csv(melody_csv, encoding='utf-8')

  if beat_df['signature'].unique().tolist() != accept_timesig:
    return []

  # --- Collect events ---
  beat_dict = collect_beats(beat_df, vocab)
  if use_structure and 'form' in beat_df.columns:
    beat_dict = collect_segments(beat_df, beat_dict)

  chords = collect_chords(beat_df, chord_processor)
  notes, _ = collect_notes(melody_df, beat_dict)
  if use_structure and mlu_processor is not None:
    notes = collect_mlus(melody_df, notes, mlu_processor)

  beats = [beat_dict[k] for k in sorted(beat_dict.keys(), key=lambda x: (x[0], x[1]))]

  # Group by bar
  events_by_bar = group_events_by_bar(beats, chords, notes)
  bars = sorted(events_by_bar.keys())

  if len(bars) < window_bars:
    return []

  # PAD token = vocab size (appended conceptually, not stored in vocab)
  pad_id = len(vocab.event2idx)

  pairs = []
  for start_i in range(0, len(bars) - window_bars + 1, stride):
    win_bars = bars[start_i : start_i + window_bars]

    enc_ids = encode_encoder_events(events_by_bar, win_bars, vocab, chord_processor, use_structure=use_structure)
    dec_ids = encode_decoder_events(events_by_bar, win_bars, vocab, use_structure=use_structure)

    enc_ids = pad_or_truncate(enc_ids, max_seq_len, pad_id)
    dec_ids = pad_or_truncate(dec_ids, max_seq_len, pad_id)

    pairs.append( (enc_ids, dec_ids) )

  return pairs


def convert_all_seq2seq(beat_csvs, melody_csvs, out_path, vocab, chord_processor,
                        window_bars=8, stride=1, max_seq_len=512,
                        use_structure=False, mlu_processor=None):
  """
  Batch-convert all pieces into seq2seq pairs and save as a single .npz file.

  Saved arrays:
    'encoder_inputs':  (N, max_seq_len) int32
    'decoder_targets': (N, max_seq_len) int32
  """
  all_enc, all_dec = [], []
  for idx, (bc, mc) in enumerate(zip(beat_csvs, melody_csvs)):
    print ('file no: {} >> {}'.format(idx + 1, mc))
    pairs = convert_piece_seq2seq(
      bc, mc, vocab, chord_processor,
      window_bars=window_bars, stride=stride, max_seq_len=max_seq_len,
      use_structure=use_structure, mlu_processor=mlu_processor
    )
    for enc, dec in pairs:
      all_enc.append(enc)
      all_dec.append(dec)

  encoder_inputs  = np.array(all_enc, dtype=np.int32)
  decoder_targets = np.array(all_dec, dtype=np.int32)

  print ('>> total pairs: {}'.format(len(all_enc)))
  print ('>> encoder shape: {}'.format(encoder_inputs.shape))
  print ('>> decoder shape: {}'.format(decoder_targets.shape))

  np.savez(out_path, encoder_inputs=encoder_inputs, decoder_targets=decoder_targets)
  print ('>> saved to: {}'.format(out_path))

  return encoder_inputs, decoder_targets


# ---------------------------------------------------------------------------
# Top-level conversion functions (original interleaved pipeline)
# ---------------------------------------------------------------------------

def convert_piece(beat_csv, melody_csv, out_csv, vocab, chord_processor, accept_timesig=['\'4/4\''], use_structure=False, mlu_processor=None):
  """
  Convert a single piece (beat CSV + melody CSV) into a REMI encoding CSV.

  Steps:
    1. Read beat and melody DataFrames.
    2. Filter out non-4/4 pieces (configurable via accept_timesig).
    3. Collect beats, optional segments, chords, notes, and optional MLU tags.
    4. Merge and sort all events chronologically.
    5. Encode events into token rows and save to out_csv.

  Returns the number of token rows written (or 0 if the piece was skipped).
  """
  beat_df   = pd.read_csv(beat_csv,   encoding='utf-8')
  melody_df = pd.read_csv(melody_csv, encoding='utf-8')
  
  # Skip pieces with unsupported time signatures
  if beat_df['signature'].unique().tolist() != accept_timesig:
    print ('[Info] Invalid time signature: {}, skipping ...'.format(beat_df['signature'].unique().tolist()))
    return 0

  # --- Collect events ---
  beat_dict = collect_beats(beat_df, vocab)
  if use_structure and 'form' in beat_df.columns:
    beat_dict = collect_segments(beat_df, beat_dict)

  chords = collect_chords(beat_df, chord_processor)
  notes, n_short = collect_notes(melody_df, beat_dict)
  if use_structure:
    notes = collect_mlus(melody_df, notes, mlu_processor)

  # Flatten beat_dict and sort by (bar, beat) to get an ordered beat list
  beats  = [beat_dict[k] for k in sorted(beat_dict.keys(), key=lambda x: (x[0], x[1]))]

  # Merge all event types and sort by musical time
  events = sorted(notes + chords + beats, key=cmp_to_key(sort_remi_events_cmp))

  df_enc = event_to_encodings(events, vocab, chord_processor, use_structure=use_structure)

  print ('>> writing to:', out_csv)
  df_enc.to_csv(out_csv, encoding='utf-8', index=False)

  return len(df_enc)


def convert_all_pieces(beat_csvs, melody_csvs, out_dir, vocab, chord_processor, use_structure=False, mlu_processor=None):
  """
  Batch-convert all pieces and save REMI CSVs to out_dir.

  Output filenames: <melody_filename>_remi.csv

  After processing, plots a histogram of token sequence lengths (capped at 6000)
  to give an overview of dataset size distribution.
  """
  if not os.path.exists(out_dir):
    os.makedirs(out_dir)

  pieces_len = []
  idx = 1
  for bc, mc in zip(beat_csvs, melody_csvs):
    print ('file no:', idx)
    idx += 1
    print ('>> now processing:', mc)
    # Build output path: same filename as melody CSV but with _remi suffix
    out_csv = os.path.join(out_dir, mc.replace('\\', '/').split('/')[-1].replace('.csv', '_remi.csv'))
    pieces_len.append( convert_piece(bc, mc, out_csv, vocab, chord_processor, use_structure=use_structure, mlu_processor=mlu_processor) )

  # Visualize the distribution of encoded sequence lengths across all pieces
  plt.hist(pieces_len, bins=20, range=(min(pieces_len), 6000), rwidth=0.6)
  plt.show()
  return


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
  # Load all beat and melody CSVs (must be sorted so they are aligned pairwise)
  beat_csvs   = sorted( glob('../mcsv_beat/*.csv') )
  melody_csvs = sorted( glob('../mcsv_melody/*.csv') )

  # Load pre-built vocabulary and processors from pickles
  vocab    = pickle.load( open('../pickles/remi_wstruct_vocab.pkl', 'rb') )
  ch_proc  = pickle.load( open('../pickles/chord_processor.pkl',   'rb') )
  mlu_proc = pickle.load( open('../pickles/mlu_processor.pkl',     'rb') )

  # Uncomment to test a single piece:
  # convert_piece(beat_csvs[317], melody_csvs[317], 'xxx', vocab, ch_proc, use_structure=True, mlu_processor=mlu_proc)

  # Convert all pieces with full structural annotation
  # convert_all_pieces(beat_csvs, melody_csvs, '../remi_encs_struct', vocab, ch_proc, use_structure=True, mlu_processor=mlu_proc)

  convert_all_seq2seq(
    beat_csvs, melody_csvs, '../seq2seq_data',
    vocab, ch_proc,
    window_bars=8, stride=1, max_seq_len=512,
    use_structure=True, mlu_processor=mlu_proc
  )