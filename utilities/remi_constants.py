import torch

REMI_VOCAB_SIZE = 470   # 469 REMI events + 1 PAD token
REMI_TOKEN_PAD  = 469   # PAD = len(vocab.event2idx)

TORCH_FLOAT      = torch.float32
TORCH_INT        = torch.int32
TORCH_LABEL_TYPE = torch.long
