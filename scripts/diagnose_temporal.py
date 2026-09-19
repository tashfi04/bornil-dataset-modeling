"""Find where temporal information is lost in the ViViT-CTC model.

The overfit check showed the model cannot memorise 200 clips: training loss
plateaus and greedy decoding stays mostly blank. CTC can only emit a sequence
if the model's output changes over time, so this measures how much each stage
varies across the time steps of a clip, compared with how much it varies
between clips. It does this for the pretrained weights and for a trained
checkpoint, on the same clips.

    python scripts/diagnose_temporal.py
    python scripts/diagnose_temporal.py --checkpoint path/to/last_checkpoint.pth
    python scripts/diagnose_temporal.py --config test --split val
"""
import os
import sys
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F


def temporal_fraction(seq):
    """Share of the variance in (N, T, F) features that comes from time.

    Near 0 means each clip's output is almost constant over time, which leaves
    CTC nothing to align. The global mean is removed first, because transformer
    features share a large common direction that would otherwise dominate.
    """
    seq = seq - seq.mean(dim=(0, 1), keepdim=True)
    within = seq.var(dim=1, unbiased=False).mean()
    between = seq.mean(dim=1).var(dim=0, unbiased=False).mean()
    total = within + between
    return (within / total).item() if total > 0 else float('nan')


def adjacent_cosine(seq):
    """Mean cosine similarity between consecutive time steps, after centering."""
    seq = seq - seq.mean(dim=(0, 1), keepdim=True)
    return F.cosine_similarity(seq[:, :-1], seq[:, 1:], dim=-1).mean().item()


def forward_stages(model, videos, lengths):
    """Run the model, returning the input to ViViT, pooled features and log-probs."""
    compressed = model.compressor(videos, lengths)                  # (B, 3, Tc, H, W)
    hidden = model.vivit(compressed.permute(0, 2, 1, 3, 4)).last_hidden_state
    batch = hidden.size(0)
    patches = hidden[:, 1:].reshape(batch, model.num_temporal, model.num_spatial, -1)
    pooled = patches.mean(dim=2)                                     # (B, Tt, D)
    log_probs = F.log_softmax(model.classifier(pooled).float(), dim=-1)

    # Pixels are downsampled spatially; the aim is to measure change over time,
    # which does not need full resolution
    small = F.adaptive_avg_pool3d(compressed.float(), (compressed.size(2), 20, 20))
    pixels = small.permute(0, 2, 1, 3, 4).reshape(batch, compressed.size(2), -1)
    return pixels, pooled, log_probs


def measure(model, batches, device, use_amp):
    pixels, pooled, log_probs = [], [], []
    with torch.no_grad(), torch.amp.autocast('cuda', enabled=use_amp):
        for videos, lengths in batches:
            p, q, r = forward_stages(model, videos.to(device), lengths.to(device))
            pixels.append(p.float().cpu())
            pooled.append(q.float().cpu())
            log_probs.append(r.float().cpu())
    pixels, pooled, log_probs = torch.cat(pixels), torch.cat(pooled), torch.cat(log_probs)

    argmax = log_probs.argmax(dim=-1)
    probs = log_probs.exp()
    distinct = [len(set(row[row != 0].tolist())) for row in argmax]
    return {
        'pixel_time': temporal_fraction(pixels),
        'pixel_adj': adjacent_cosine(pixels),
        'pooled_time': temporal_fraction(pooled),
        'pooled_adj': adjacent_cosine(pooled),
        'logit_time': temporal_fraction(log_probs),
        'blank_argmax': (argmax == 0).float().mean().item(),
        'blank_prob': probs[..., 0].mean().item(),
        'top_token_prob': probs[..., 1:].max(dim=-1).values.mean().item(),
        'distinct_tokens': sum(distinct) / len(distinct),
        'steps': log_probs.size(1),
        'clips': log_probs.size(0),
    }


def report(label, m):
    print(f"\n=== {label} ({m['clips']} clips, {m['steps']} CTC steps) ===")
    print("  share of variance from change over time (0 = constant in time)")
    print(f"    compressor output (input to ViViT): {m['pixel_time']:.3f}")
    print(f"    pooled ViViT features:              {m['pooled_time']:.3f}")
    print(f"    CTC log-probs:                      {m['logit_time']:.3f}")
    print("  cosine between consecutive steps (1 = no change)")
    print(f"    compressor output:                  {m['pixel_adj']:.3f}")
    print(f"    pooled ViViT features:              {m['pooled_adj']:.3f}")
    print("  CTC head output")
    print(f"    steps where blank is the argmax:    {100 * m['blank_argmax']:.1f}%")
    print(f"    mean blank probability:             {m['blank_prob']:.3f}")
    print(f"    mean top non-blank probability:     {m['top_token_prob']:.3f}")
    print(f"    distinct non-blank tokens per clip: {m['distinct_tokens']:.2f}")


def main():
    parser = argparse.ArgumentParser(description="Locate temporal information loss")
    parser.add_argument('--config', choices=['overfit', 'test', 'prod'], default='overfit')
    parser.add_argument('--split', choices=['train', 'val'], default='train')
    parser.add_argument('--checkpoint', default=None,
                        help="Trained checkpoint (defaults to last_checkpoint.pth "
                             "in the config's checkpoint directory)")
    parser.add_argument('--batches', type=int, default=8,
                        help="Batches of clips to measure")
    args = parser.parse_args()

    if args.config == 'overfit':
        from configs.overfit_vivit_ctc_config import config
    elif args.config == 'test':
        from configs.test_vivit_ctc_config import config
    else:
        from configs.vivit_ctc_config import config
    from src.training.vivit_trainer import ViViTTrainer

    config.auto_resume = False
    trainer = ViViTTrainer(config)
    model = getattr(trainer.model, 'module', trainer.model)
    model.eval()
    device = config.device
    use_amp = device.type == 'cuda'

    checkpoint_path = args.checkpoint or os.path.join(
        trainer.checkpoint_dir, 'last_checkpoint.pth')
    if not os.path.exists(checkpoint_path):
        print(f"No checkpoint at {checkpoint_path}; pass --checkpoint")
        sys.exit(1)

    # The same clips are measured before and after loading the checkpoint
    loader = trainer.train_loader if args.split == 'train' else trainer.val_loader
    batches = []
    for index, batch in enumerate(loader):
        if index >= args.batches:
            break
        batches.append((batch['videos'], batch['video_lengths']))

    before = measure(model, batches, device, use_amp)
    report("Pretrained ViViT (CTC head untrained)", before)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    after = measure(model, batches, device, use_amp)
    report(f"Trained checkpoint, epoch {checkpoint.get('epoch', '?')}", after)

    print("\n=== Reading it ===")
    print("Compare the three 'change over time' numbers after training. Where the")
    print("value drops sharply from one stage to the next is where temporal")
    print("information is being lost:")
    print("  - low already at the compressor output: the input to ViViT barely")
    print("    changes over time (averaging, or clips that are mostly static)")
    print("  - high at the compressor, low after pooling: ViViT plus spatial")
    print("    mean-pooling is flattening time")
    print("  - high in the pooled features but blank dominating the head: the")
    print("    features vary, and the problem is the CTC head or its time budget")


if __name__ == "__main__":
    main()
