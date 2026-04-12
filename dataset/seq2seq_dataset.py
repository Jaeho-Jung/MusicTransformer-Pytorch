import numpy as np
import torch
from torch.utils.data import Dataset

from utilities.remi_constants import REMI_TOKEN_PAD, TORCH_LABEL_TYPE


class Seq2SeqDataset(Dataset):
    """
    Dataset for seq2seq music generation from .npz file.

    The .npz contains:
        encoder_inputs:  (N, max_seq_len) int32  — harmony tokens
        decoder_targets: (N, max_seq_len) int32  — melody tokens

    For teacher forcing, decoder_input is decoder_target shifted right by 1,
    with a PAD token prepended at position 0.
    """

    def __init__(self, npz_path, split='train', train_ratio=0.9, seed=42, augment=False, pitchaug_range=(-3, 3)):
        data = np.load(npz_path)
        enc = data['encoder_inputs']
        dec = data['decoder_targets']

        # Reproducible train/val split
        n = len(enc)
        rng = np.random.RandomState(seed)
        indices = rng.permutation(n)
        split_idx = int(n * train_ratio)

        if split == 'train':
            indices = indices[:split_idx]
        elif split == 'val':
            indices = indices[split_idx:]
            augment = False  # Never augment validation data
        else:
            raise ValueError("split must be 'train' or 'val'")

        self.encoder_inputs  = enc[indices]
        self.decoder_targets = dec[indices]
        self.augment = augment
        self.pitchaug_range = pitchaug_range

    def __len__(self):
        return len(self.encoder_inputs)

    def _augment_pitch(self, seq, shift):
        if shift == 0:
            return seq

        # Note-On IDs: 0 to 127
        note_on_mask = (seq >= 0) & (seq <= 127)

        # Chord-Tone IDs: 322 (C) to 333 (B)
        chord_tone_mask = (seq >= 322) & (seq <= 333)

        # Chord-Slash IDs: 334 (C) to 345 (B)
        chord_slash_mask = (seq >= 334) & (seq <= 345)

        # Apply shifts
        # 1. Note-On: simple shift, but clip to 0-127
        seq[note_on_mask] = torch.clamp(seq[note_on_mask] + shift, min=0, max=127)

        # 2. Chord-Tone: circular shift within 322-333
        seq[chord_tone_mask] = 322 + ((seq[chord_tone_mask] - 322 + shift) % 12)

        # 3. Chord-Slash: circular shift within 334-345
        seq[chord_slash_mask] = 334 + ((seq[chord_slash_mask] - 334 + shift) % 12)

        return seq

    def __getitem__(self, idx):
        enc = torch.tensor(self.encoder_inputs[idx], dtype=TORCH_LABEL_TYPE)
        dec_target = torch.tensor(self.decoder_targets[idx], dtype=TORCH_LABEL_TYPE)

        if self.augment:
            shift = torch.randint(self.pitchaug_range[0], self.pitchaug_range[1] + 1, (1,)).item()
            enc = self._augment_pitch(enc, shift)
            dec_target = self._augment_pitch(dec_target, shift)

        # Shift right: prepend PAD, remove last token
        dec_input = torch.cat([
            torch.tensor([REMI_TOKEN_PAD], dtype=TORCH_LABEL_TYPE),
            dec_target[:-1]
        ])

        return enc, dec_input, dec_target
