import torch
import torch.nn as nn
from torch.nn.modules.normalization import LayerNorm

from utilities.remi_constants import *
from utilities.device import get_device

from .positional_encoding import PositionalEncoding


class MusicTransformerBaseline(nn.Module):
    """
    Standard Transformer encoder-decoder for conditional melody generation.
    Uses PyTorch built-in multi-head attention (no RPR) as a baseline
    for comparison against the RPR-based MusicTransformerSeq2Seq.
    """

    def __init__(self, n_layers=6, num_heads=8, d_model=512, dim_feedforward=1024,
                 dropout=0.1, max_sequence=512, vocab_size=REMI_VOCAB_SIZE,
                 pad_token=REMI_TOKEN_PAD):
        super(MusicTransformerBaseline, self).__init__()

        self.nlayers    = n_layers
        self.nhead      = num_heads
        self.d_model    = d_model
        self.d_ff       = dim_feedforward
        self.dropout    = dropout
        self.max_seq    = max_sequence
        self.vocab_size = vocab_size
        self.pad_token  = pad_token

        # Shared embedding (encoder and decoder use same vocab)
        self.embedding = nn.Embedding(self.vocab_size, self.d_model, padding_idx=self.pad_token)

        # Positional encoding (sinusoidal, same as RPR model)
        self.positional_encoding = PositionalEncoding(self.d_model, self.dropout, self.max_seq)

        # Encoder (standard multi-head attention)
        encoder_norm = LayerNorm(self.d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            self.d_model, self.nhead, self.d_ff, self.dropout, batch_first=False
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, self.nlayers, encoder_norm)

        # Decoder (standard multi-head attention)
        decoder_norm = LayerNorm(self.d_model)
        decoder_layer = nn.TransformerDecoderLayer(
            self.d_model, self.nhead, self.d_ff, self.dropout, batch_first=False
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, self.nlayers, decoder_norm)

        # Output projection
        self.Wout = nn.Linear(self.d_model, self.vocab_size)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None,
                src_key_padding_mask=None, tgt_key_padding_mask=None,
                memory_key_padding_mask=None):
        """
        Args:
            src: encoder input, (batch, src_len) int
            tgt: decoder input, (batch, tgt_len) int
            src_mask: optional, (src_len, src_len)
            tgt_mask: causal mask, (tgt_len, tgt_len)
            src_key_padding_mask: (batch, src_len) bool — True for PAD positions
            tgt_key_padding_mask: (batch, tgt_len) bool — True for PAD positions
            memory_key_padding_mask: same as src_key_padding_mask

        Returns:
            logits: (batch, tgt_len, vocab_size)
        """
        # Generate causal mask for decoder if not provided
        if tgt_mask is None:
            tgt_mask = self.generate_square_subsequent_mask(tgt.shape[1]).to(get_device())

        # Encoder
        src_emb = self.embedding(src)                   # (B, S, d_model)
        src_emb = src_emb.permute(1, 0, 2)              # (S, B, d_model)
        src_emb = self.positional_encoding(src_emb)
        memory = self.encoder(src_emb, mask=src_mask,
                              src_key_padding_mask=src_key_padding_mask)

        # Decoder
        tgt_emb = self.embedding(tgt)                   # (B, T, d_model)
        tgt_emb = tgt_emb.permute(1, 0, 2)              # (T, B, d_model)
        tgt_emb = self.positional_encoding(tgt_emb)
        output = self.decoder(tgt_emb, memory, tgt_mask=tgt_mask,
                              tgt_key_padding_mask=tgt_key_padding_mask,
                              memory_key_padding_mask=memory_key_padding_mask)

        # Back to (B, T, d_model)
        output = output.permute(1, 0, 2)

        # Project to vocab
        logits = self.Wout(output)

        del tgt_mask
        return logits

    def generate_square_subsequent_mask(self, sz):
        """Generate causal mask: positions can only attend to earlier positions."""
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
