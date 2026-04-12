"""
Train the encoder-decoder Music Transformer on seq2seq REMI data.

Usage:
    python train_seq2seq.py --data seq2seq_data/seq2seq_data.npz --output_dir output/seq2seq
"""
import os
import csv
import argparse
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from dataset.seq2seq_dataset import Seq2SeqDataset
from model.music_transformer_seq2seq import MusicTransformerSeq2Seq
from model.loss import SmoothCrossEntropyLoss

from utilities.remi_constants import REMI_VOCAB_SIZE, REMI_TOKEN_PAD
from utilities.device import get_device, use_cuda
from utilities.lr_scheduling import LrStepTracker, get_lr


def parse_args():
    parser = argparse.ArgumentParser(description='Train seq2seq Music Transformer')
    parser.add_argument('--data', type=str, required=True, help='Path to .npz data file')
    parser.add_argument('--output_dir', type=str, default='output/seq2seq', help='Output directory')

    # Model
    parser.add_argument('--n_layers', type=int, default=6)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--dim_feedforward', type=int, default=1024)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--max_sequence', type=int, default=512)

    # Training
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=None, help='Static LR (None = warmup scheduler)')
    parser.add_argument('--ce_smoothing', type=float, default=None, help='Label smoothing (None = standard CE)')
    parser.add_argument('--warmup_steps', type=int, default=4000)
    parser.add_argument('--n_workers', type=int, default=4)
    parser.add_argument('--weight_modulus', type=int, default=5, help='Save weights every N epochs')
    parser.add_argument('--print_modulus', type=int, default=50, help='Print loss every N batches')

    # Misc
    parser.add_argument('--force_cpu', action='store_true')
    parser.add_argument('--resume', action='store_true', help='Resume from latest checkpoint if available')
    parser.add_argument('--augment', action='store_true', help='Enable random pitch augmentation [-3, +3]')
    return parser.parse_args()


def train_epoch(epoch, model, dataloader, loss_func, opt, lr_scheduler, print_modulus):
    import time
    model.train()
    total_loss = 0.0
    n_batches = 0
    start_time = time.time()

    for batch_idx, (enc, dec_input, dec_target) in enumerate(dataloader):
        enc = enc.to(get_device())
        dec_input = dec_input.to(get_device())
        dec_target = dec_target.to(get_device())

        # Padding masks (True where PAD)
        src_pad_mask = (enc == REMI_TOKEN_PAD)
        tgt_pad_mask = (dec_target == REMI_TOKEN_PAD)

        # Forward
        logits = model(enc, dec_input,
                       src_key_padding_mask=src_pad_mask,
                       tgt_key_padding_mask=tgt_pad_mask,
                       memory_key_padding_mask=src_pad_mask)

        # Loss: flatten (B, T, V) -> (B*T, V), target (B, T) -> (B*T)
        logits_flat = logits.reshape(-1, logits.shape[-1])
        target_flat = dec_target.reshape(-1)

        loss = loss_func(logits_flat, target_flat)

        # Backward
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        total_loss += loss.item()
        n_batches += 1

        if (batch_idx + 1) % print_modulus == 0:
            elapsed = time.time() - start_time
            avg_time = elapsed / n_batches
            rem_batches = len(dataloader) - n_batches
            eta_seconds = int(rem_batches * avg_time)
            if eta_seconds > 60:
                eta_str = f"{eta_seconds // 60}m {eta_seconds % 60}s"
            else:
                eta_str = f"{eta_seconds}s"

            print('  Epoch {} | Batch {}/{} | Loss: {:.4f} | LR: {:.6f} | ETA: {}'.format(
                epoch, batch_idx + 1, len(dataloader), loss.item(), get_lr(opt), eta_str))

    return total_loss / max(n_batches, 1)


def eval_model(model, dataloader, loss_func):
    model.eval()
    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for enc, dec_input, dec_target in dataloader:
            enc = enc.to(get_device())
            dec_input = dec_input.to(get_device())
            dec_target = dec_target.to(get_device())

            src_pad_mask = (enc == REMI_TOKEN_PAD)
            tgt_pad_mask = (dec_target == REMI_TOKEN_PAD)

            logits = model(enc, dec_input,
                           src_key_padding_mask=src_pad_mask,
                           tgt_key_padding_mask=tgt_pad_mask,
                           memory_key_padding_mask=src_pad_mask)

            logits_flat = logits.reshape(-1, logits.shape[-1])
            target_flat = dec_target.reshape(-1)

            loss = loss_func(logits_flat, target_flat)
            total_loss += loss.item()
            n_batches += 1

    return total_loss / max(n_batches, 1)


def main():
    args = parse_args()

    if args.force_cpu:
        use_cuda(False)
        print('WARNING: Forced CPU usage')

    os.makedirs(args.output_dir, exist_ok=True)
    weights_dir = os.path.join(args.output_dir, 'weights')
    results_dir = os.path.join(args.output_dir, 'results')
    os.makedirs(weights_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    results_file = os.path.join(results_dir, 'results.csv')
    best_loss_file = os.path.join(results_dir, 'best_loss_weights.pt')
    checkpoint_file = os.path.join(args.output_dir, 'latest_checkpoint.pt')

    # Datasets
    print('Loading data from:', args.data)
    train_dataset = Seq2SeqDataset(args.data, split='train', augment=args.augment)
    val_dataset   = Seq2SeqDataset(args.data, split='val', augment=False)
    print('Train: {} pairs | Val: {} pairs'.format(len(train_dataset), len(val_dataset)))

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              num_workers=args.n_workers, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=args.batch_size,
                              num_workers=args.n_workers)

    # Model
    model = MusicTransformerSeq2Seq(
        n_layers=args.n_layers, num_heads=args.num_heads,
        d_model=args.d_model, dim_feedforward=args.dim_feedforward,
        dropout=args.dropout, max_sequence=args.max_sequence
    ).to(get_device())

    n_params = sum(p.numel() for p in model.parameters())
    print('Model parameters: {:,}'.format(n_params))

    # Loss
    eval_loss_func = nn.CrossEntropyLoss(ignore_index=REMI_TOKEN_PAD)
    if args.ce_smoothing is not None:
        train_loss_func = SmoothCrossEntropyLoss(args.ce_smoothing, REMI_VOCAB_SIZE, ignore_index=REMI_TOKEN_PAD)
    else:
        train_loss_func = eval_loss_func

    # Optimizer + scheduler
    start_epoch = 0
    best_val_loss = float('inf')

    if args.lr is None:
        lr = 1.0
        opt = Adam(model.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9)
        lr_stepper = LrStepTracker(args.d_model, args.warmup_steps, 0)
        lr_scheduler = LambdaLR(opt, lr_stepper.step)
    else:
        opt = Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-9)
        lr_scheduler = None

    # Resume from checkpoint
    if args.resume and os.path.isfile(checkpoint_file):
        ckpt = torch.load(checkpoint_file, map_location=get_device())
        model.load_state_dict(ckpt['model_state_dict'])
        opt.load_state_dict(ckpt['optimizer_state_dict'])
        if lr_scheduler is not None and 'scheduler_state_dict' in ckpt:
            lr_scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        start_epoch = ckpt['epoch']
        best_val_loss = ckpt['best_val_loss']
        print('Resumed from epoch {} (best val loss: {:.4f})'.format(start_epoch, best_val_loss))
    elif args.resume:
        print('No checkpoint found, starting from scratch')

    # Results CSV
    if not os.path.isfile(results_file):
        with open(results_file, 'w', newline='') as f:
            csv.writer(f).writerow(['Epoch', 'LR', 'Train_Loss', 'Val_Loss'])
    for epoch in range(start_epoch, args.epochs):
        print('=' * 50)
        print('Epoch {}/{}'.format(epoch + 1, args.epochs))
        print('=' * 50)

        train_loss = train_epoch(epoch + 1, model, train_loader, train_loss_func,
                                 opt, lr_scheduler, args.print_modulus)
        val_loss = eval_model(model, val_loader, eval_loss_func)

        lr_now = get_lr(opt)
        print('Train loss: {:.4f} | Val loss: {:.4f} | LR: {:.6f}'.format(
            train_loss, val_loss, lr_now))

        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_loss_file)
            print('>> New best val loss: {:.4f}'.format(val_loss))

        # Periodic save (model weights only)
        if (epoch + 1) % args.weight_modulus == 0:
            path = os.path.join(weights_dir, 'epoch_{:04d}.pt'.format(epoch + 1))
            torch.save(model.state_dict(), path)

        # Full checkpoint for resume (saved every epoch)
        ckpt = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': opt.state_dict(),
            'best_val_loss': best_val_loss,
        }
        if lr_scheduler is not None:
            ckpt['scheduler_state_dict'] = lr_scheduler.state_dict()
        torch.save(ckpt, checkpoint_file)

        # Log
        with open(results_file, 'a', newline='') as f:
            csv.writer(f).writerow([epoch + 1, lr_now, train_loss, val_loss])


if __name__ == '__main__':
    main()
