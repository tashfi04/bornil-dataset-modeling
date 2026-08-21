"""Check the ViViT-CTC pipeline without running a training job.

Covers:
  1. Config arithmetic, so the CTC time axis can fit `max_bpe_tokens`
  2. Model build, including position embedding interpolation
  3. Forward pass shape
  4. CTC loss is finite and gradients reach the temporal compressor
  5. Frame cache encode/decode round-trip
  6. Peak VRAM, for sizing batch_size

Run on the machine intended for training:

    python scripts/smoke_test_vivit.py
    python scripts/smoke_test_vivit.py --config test    # Kaggle test config
    python scripts/smoke_test_vivit.py --batch-size 8   # probe a larger batch
"""
import os
import sys
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np

PASS = "  [PASS]"
FAIL = "  [FAIL]"


def main():
    parser = argparse.ArgumentParser(description="Smoke test the ViViT-CTC pipeline")
    parser.add_argument('--config', choices=['prod', 'test'], default='prod')
    parser.add_argument('--batch-size', type=int, default=None,
                        help="Override batch size (use to probe VRAM headroom)")
    args = parser.parse_args()

    if args.config == 'test':
        from configs.test_vivit_ctc_config import config
    else:
        from configs.vivit_ctc_config import config

    from src.models.vivit_ctc_hf import ViViT_CTC_HF, interpolate_position_grid
    from src.utils.frame_cache import encode_frames, decode_frames

    failures = []

    def check(label, condition, detail=""):
        print(f"{PASS if condition else FAIL} {label}" + (f" - {detail}" if detail else ""))
        if not condition:
            failures.append(label)

    batch_size = args.batch_size or config.batch_size
    device = config.device
    print(f"Config: {args.config} ({getattr(config, 'model_type', '?')})")
    if device.type == 'cuda':
        props = torch.cuda.get_device_properties(0)
        print(f"GPU: {props.name}, {props.total_memory / 1024 ** 3:.2f} GB, "
              f"{torch.cuda.device_count()} visible")

    # ---------------------------------------------------- 1. config
    print("\n=== 1. Config arithmetic ===")
    num_frames = config.num_frames
    compressed = getattr(config, 'compressed_frames', num_frames)
    frame_h, frame_w = config.frame_size
    tubelet_t = 2  # confirmed against the loaded model below
    ctc_steps = compressed // tubelet_t

    print(f"  frames read={num_frames}  compressed={compressed}  frame_size={config.frame_size}")
    print(f"  expected CTC steps = {compressed}//{tubelet_t} = {ctc_steps}")
    print(f"  max_bpe_tokens = {config.max_bpe_tokens}")

    check("frames are square", frame_h == frame_w, f"{config.frame_size}")
    check("frame size divisible by 16", frame_h % 16 == 0)
    check("compressed_frames divisible by tubelet", compressed % tubelet_t == 0)
    check("max_bpe_tokens fits the CTC axis",
          config.max_bpe_tokens <= ctc_steps,
          f"need max_bpe_tokens <= {ctc_steps}, got {config.max_bpe_tokens}")

    # ---------------------------------------------------- 2. build
    print("\n=== 2. Model build ===")
    num_classes = getattr(config, 'bpe_vocab_size', 2000) + 1
    model = ViViT_CTC_HF(config=config, num_classes=num_classes).to(device)

    # from_pretrained leaves the model in eval mode, and HF only applies gradient
    # checkpointing while training. Match what the trainer does.
    model.train()

    tubelet_t = model.vivit.config.tubelet_size[0]
    expected_steps = compressed // tubelet_t
    pos_len = model.vivit.embeddings.position_embeddings.shape[1]
    expected_pos = model.num_temporal * model.num_spatial + 1

    print(f"  tubelet_size = {model.vivit.config.tubelet_size}")
    print(f"  temporal tokens = {model.num_temporal}, spatial tokens = {model.num_spatial}")
    print(f"  total ViViT tokens = {model.num_temporal * model.num_spatial}")
    check("position embeddings resized", pos_len == expected_pos,
          f"{pos_len} vs expected {expected_pos}")
    check("model output_length matches config", model.output_length == expected_steps,
          f"{model.output_length} vs {expected_steps}")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  parameters: {total:,} total, {trainable:,} trainable")

    # Must hold even when the backbone is frozen
    comp_trainable = all(p.requires_grad for p in model.compressor.parameters())
    check("temporal compressor is trainable", comp_trainable)

    # ---------------------------------------------- 2b. interpolation correctness
    print("\n=== 2b. Position-embedding interpolation ===")

    # Axis ordering is the easy thing to get wrong here, and a transposed axis
    # would still produce the right shape. Encode each position's own indices
    # into a synthetic grid and check they survive the round trip in order.
    old_shape, new_shape = (4, 3, 3), (8, 6, 6)
    ot, oh, ow = old_shape
    probe = torch.zeros(1, ot * oh * ow, 3)
    for t in range(ot):
        for h in range(oh):
            for w in range(ow):
                probe[0, t * oh * ow + h * ow + w] = torch.tensor(
                    [float(t), float(h), float(w)]
                )
    out = interpolate_position_grid(probe, old_shape, new_shape).reshape(*new_shape, 3)

    # Channel 0 encodes time, 1 height, 2 width. Each must increase only along
    # its own axis and stay flat along the others.
    t_ok = bool((out[1:, :, :, 0] > out[:-1, :, :, 0]).all())
    h_ok = bool((out[:, 1:, :, 1] > out[:, :-1, :, 1]).all())
    w_ok = bool((out[:, :, 1:, 2] > out[:, :, :-1, 2]).all())
    t_flat = out[:, :, :, 0].std(dim=(1, 2)).max().item()
    h_flat = out[:, :, :, 1].std(dim=2).max().item()
    check("time axis maps to time", t_ok)
    check("height axis maps to height", h_ok)
    check("width axis maps to width", w_ok)
    check("axes stay independent", max(t_flat, h_flat) < 1e-5,
          f"max cross-axis spread {max(t_flat, h_flat):.2e}")
    check("identity when shape is unchanged",
          torch.equal(interpolate_position_grid(probe, old_shape, old_shape), probe))

    # The real interpolated grid should still be locally smooth: neighbouring
    # positions closer to each other than randomly chosen pairs.
    with torch.no_grad():
        pos = model.vivit.embeddings.position_embeddings.detach().float().cpu()
        gt, gh, gw = model.position_grid_shape
        grid = pos[0, 1:].reshape(gt, gh, gw, -1)

        def mean_dist(a, b):
            return (a - b).pow(2).sum(-1).sqrt().mean().item()

        temporal = mean_dist(grid[1:], grid[:-1])
        spatial = mean_dist(grid[:, 1:], grid[:, :-1])
        flat = grid.reshape(-1, grid.size(-1))
        perm = torch.randperm(flat.size(0))
        random_pairs = mean_dist(flat, flat[perm])

    print(f"  mean distance: temporal-adjacent={temporal:.4f} "
          f"spatial-adjacent={spatial:.4f} random={random_pairs:.4f}")
    check("neighbouring positions remain similar",
          temporal < random_pairs and spatial < random_pairs,
          "adjacent positions should be closer than random ones")

    # ---------------------------------------------------- 3. compressor identity
    print("\n=== 3. Compressor initialization ===")
    with torch.no_grad():
        probe = torch.rand(1, 3, num_frames, 8, 8)
        out = model.compressor.to('cpu')(probe)
        expected_avg = torch.nn.functional.adaptive_avg_pool3d(probe, (compressed, 8, 8))
        max_dev = (out - expected_avg).abs().max().item()
    model.compressor.to(device)
    check("initialized to exact temporal average pooling", max_dev < 1e-4,
          f"max deviation {max_dev:.2e}")
    print(f"  compressor output shape: {tuple(out.shape)}")

    # ------------------------------------------- 3b. variable-length handling
    print("\n=== 3b. Padded-batch handling ===")

    # collate_fn zero-pads a batch to its longest clip. Mark real frames as 1.0
    # and padding as 0.0: if the padding leaked into the resampling, the shorter
    # clips would come out darker in proportion to how much was padded.
    lengths = [21, 64, 100, num_frames]
    padded = torch.zeros(len(lengths), 3, num_frames, 4, 4)
    for i, n in enumerate(lengths):
        padded[i, :, :n] = 1.0

    compressor = model.compressor.to('cpu')
    with torch.no_grad():
        aware = compressor(padded, torch.tensor(lengths))
        naive = compressor(padded, None)
    model.compressor.to(device)

    print(f"  {'real frames':>12} {'length-aware':>14} {'ignoring lengths':>18}")
    worst = 0.0
    for i, n in enumerate(lengths):
        got, ignored = aware[i].mean().item(), naive[i].mean().item()
        worst = max(worst, abs(got - 1.0))
        print(f"  {n:>12} {got:>14.4f} {ignored:>18.4f}")
    check("padding never enters the resampling", worst < 1e-4,
          f"worst deviation from 1.0 is {worst:.2e}")
    check("output length is uniform regardless of input length",
          tuple(aware.shape) == (len(lengths), 3, compressed, 4, 4),
          f"{tuple(aware.shape)}")

    # ---------------------------------------------------- 4. forward/back
    print(f"\n=== 4. Forward + backward (batch_size={batch_size}) ===")
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()

    videos = torch.rand(batch_size, 3, num_frames, frame_h, frame_w, device=device)
    video_lengths = torch.full((batch_size,), num_frames, dtype=torch.long, device=device)

    use_amp = bool(getattr(config, 'use_amp', False)) and device.type == 'cuda'
    try:
        with torch.amp.autocast('cuda', enabled=use_amp):
            outputs = model(videos, video_lengths)
    except torch.OutOfMemoryError:
        tokens = model.num_temporal * model.num_spatial
        print(f"{FAIL} forward pass ran out of memory at batch_size={batch_size}")
        print(f"  This config needs {tokens} ViViT tokens per sample.")
        print(f"  Options, cheapest first:")
        print(f"    - lower batch_size (raise gradient_accumulation_steps to compensate)")
        print(f"    - set frame_size to (160, 160): {tokens} -> "
              f"{model.num_temporal * (frame_h // 16) ** 2 // 4} tokens")
        print(f"    - lower compressed_frames (also shortens the CTC axis, so "
              f"max_bpe_tokens must drop with it)")
        print(f"    - confirm gradient_checkpointing is True in the config")
        sys.exit(1)

    # Models return batch-first so nn.DataParallel gathers replicas correctly
    print(f"  output shape: {tuple(outputs.shape)}  (expect ({batch_size}, {expected_steps}, {num_classes}))")
    check("output shape correct",
          tuple(outputs.shape) == (batch_size, expected_steps, num_classes))

    # Longest target the filter still allows through
    target_len = min(config.max_bpe_tokens, expected_steps)
    text_lengths = torch.full((batch_size,), target_len, dtype=torch.long)
    text_targets = torch.randint(1, num_classes, (batch_size * target_len,), dtype=torch.long)
    input_lengths = torch.full((batch_size,), outputs.size(1), dtype=torch.long)

    criterion = torch.nn.CTCLoss(blank=0, zero_infinity=True)
    log_probs = outputs.permute(1, 0, 2).float().cpu()   # CTC wants (T, B, C)
    loss = criterion(log_probs, text_targets, input_lengths, text_lengths)
    print(f"  CTC loss (target_len={target_len}): {loss.item():.4f}")
    check("CTC loss is finite and non-zero",
          torch.isfinite(loss) and loss.item() > 0,
          "zero/inf loss means input_length < target_length somewhere")

    loss.backward()
    comp_grad = model.compressor.temporal_conv.weight.grad
    check("gradient reaches the compressor through frozen ViViT",
          comp_grad is not None and torch.isfinite(comp_grad).all()
          and comp_grad.abs().sum().item() > 0)

    if device.type == 'cuda':
        peak = torch.cuda.max_memory_allocated() / 1024 ** 3
        print(f"  peak VRAM: {peak:.2f} GB at batch_size={batch_size}")
        print(f"  gradient_checkpointing = {getattr(config, 'gradient_checkpointing', False)}, use_amp = {use_amp}")

    # ---------------------------------------------------- 5. frame cache
    print("\n=== 5. Frame cache round-trip ===")
    try:
        frames = [np.random.randint(0, 255, (frame_h, frame_w, 3), dtype=np.uint8)
                  for _ in range(8)]
        blob = encode_frames(frames, quality=90)
        restored = decode_frames(blob)
        check("round-trip preserves count and shape",
              len(restored) == 8 and restored[0].shape == frames[0].shape)
        print(f"  {len(frames)} frames -> {len(blob)/1024:.1f} KB "
              f"({len(blob)/len(frames)/1024:.1f} KB/frame at {frame_h}px)")
        projected = len(blob) / len(frames) * num_frames * 17988 / 1024 ** 3
        print(f"  NOTE: random noise is worst-case for JPEG. Real footage will be")
        print(f"  far smaller than this {projected:.1f} GB projection - use")
        print(f"  scripts/preprocess_cache_frames.py --dry-run for a real number.")
    except Exception as exc:
        check("frame cache round-trip", False, f"{type(exc).__name__}: {exc}")

    # ---------------------------------------------------- summary
    print("\n" + "=" * 60)
    if failures:
        print(f"FAILED {len(failures)} check(s):")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    print("All checks passed.")
    print(f"CTC axis = {expected_steps} steps, max_bpe_tokens = {config.max_bpe_tokens}")


if __name__ == "__main__":
    main()
