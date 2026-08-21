import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import VivitModel


def interpolate_position_grid(grid_pos, old_shape, new_shape):
    """Resize a flattened (time, height, width) position-embedding grid.

    grid_pos is (1, old_t*old_h*old_w, dim) with time slowest and width fastest,
    matching how HF flattens the tubelet conv output. Returns the same layout at
    new_shape.
    """
    old_t, old_h, old_w = old_shape
    new_t, new_h, new_w = new_shape
    dim = grid_pos.size(-1)

    expected = old_t * old_h * old_w
    if grid_pos.size(1) != expected:
        raise ValueError(
            f"grid has {grid_pos.size(1)} positions, expected {expected} "
            f"for {old_t}x{old_h}x{old_w}"
        )
    if (new_t, new_h, new_w) == (old_t, old_h, old_w):
        return grid_pos

    grid = grid_pos.reshape(1, old_t, old_h, old_w, dim).permute(0, 4, 1, 2, 3)
    grid = F.interpolate(
        grid, size=(new_t, new_h, new_w), mode='trilinear', align_corners=False
    )
    return grid.permute(0, 2, 3, 4, 1).reshape(1, new_t * new_h * new_w, dim)


class TemporalCompressor(nn.Module):
    """Resamples a frame sequence to `target_frames`, in pixel space.

    Compression happens before ViViT and keeps the 3-channel image format, so
    ViViT's pretrained tubelet embedding still receives images. The depthwise
    conv starts as an identity tap and the resampling is area-averaging, so at
    initialization the output is exactly the temporal average pool of the input:
    natural motion-blurred frames rather than noise. The conv then learns how to
    weight neighbouring frames.

    Clips shorter than target_frames are stretched rather than padded, and each
    clip in a batch is resampled over its own real length, so the zero padding
    added by collate_fn never enters the average.
    """

    def __init__(self, target_frames, kernel_size=5, channels=3):
        super().__init__()
        self.target_frames = target_frames
        self.temporal_conv = nn.Conv3d(
            channels, channels,
            kernel_size=(kernel_size, 1, 1),
            padding=(kernel_size // 2, 0, 0),
            groups=channels,  # depthwise: mixes along time only, never across RGB
            bias=False,
        )
        with torch.no_grad():
            self.temporal_conv.weight.zero_()
            self.temporal_conv.weight[:, :, kernel_size // 2] = 1.0

    def _resample(self, x):
        """Resample the time axis of one clip to exactly target_frames."""
        length = x.size(2)
        if length == self.target_frames:
            return x
        if length > self.target_frames:
            # Area averaging, so no input frame is dropped outright.
            # Spatially a no-op since H and W are unchanged.
            return F.adaptive_avg_pool3d(
                x, (self.target_frames, x.size(3), x.size(4))
            )
        # Shorter than the target: stretch the motion over the full window.
        # Linear in time beats repeating frames, and beats padding with black.
        return F.interpolate(
            x, size=(self.target_frames, x.size(3), x.size(4)),
            mode='trilinear', align_corners=False,
        )

    def forward(self, x, video_lengths=None):
        # x: (B, C, T_in, H, W)
        x = self.temporal_conv(x)

        # Batches are padded to their longest clip. Resampling across the padding
        # would average blank frames into every output frame, so each clip is
        # resampled over its own real length instead.
        if video_lengths is None:
            return self._resample(x)

        lengths = [max(1, int(n)) for n in video_lengths.tolist()]
        if all(n == x.size(2) for n in lengths):
            return self._resample(x)

        return torch.cat(
            [self._resample(x[i:i + 1, :, :n]) for i, n in enumerate(lengths)],
            dim=0,
        )


class ViViT_CTC_HF(nn.Module):
    """Pretrained HuggingFace ViViT with a CTC head, for long sign-language clips.

    Pipeline: (B, C, T_in, H, W) raw frames
      -> TemporalCompressor          T_in -> compressed_frames
      -> pretrained ViViT            position embeddings interpolated to our grid
      -> mean-pool over spatial patches
      -> linear CTC head over the temporal axis, giving (B, T, num_classes)

    The CTC time axis is `compressed_frames // tubelet_t`, which for the 16x2
    checkpoint is `compressed_frames // 2`. That value bounds how many target
    tokens a sample may have, hence `config.max_bpe_tokens`.
    """

    def __init__(self, config, num_classes):
        super().__init__()
        self.config = config
        self.num_classes = num_classes

        model_name = getattr(config, 'vivit_model_name', 'google/vivit-b-16x2-kinetics400')
        self.vivit = VivitModel.from_pretrained(model_name)

        frame_h, frame_w = config.frame_size
        if frame_h != frame_w:
            raise ValueError(
                f"ViViT uses a single image_size, so frames must be square; got {config.frame_size}"
            )

        self.compressed_frames = getattr(
            config, 'compressed_frames', getattr(config, 'num_frames', 32)
        )
        self.compressor = TemporalCompressor(self.compressed_frames)

        t_patch, h_patch, w_patch = self.vivit.config.tubelet_size
        if self.compressed_frames % t_patch:
            raise ValueError(
                f"compressed_frames ({self.compressed_frames}) must be divisible by "
                f"the tubelet temporal size ({t_patch})"
            )
        if frame_h % h_patch or frame_w % w_patch:
            raise ValueError(
                f"frame_size {config.frame_size} must be divisible by the tubelet "
                f"spatial size ({h_patch}, {w_patch})"
            )

        self.num_temporal = self.compressed_frames // t_patch
        self.num_spatial = (frame_h // h_patch) * (frame_w // w_patch)

        # CTC requires input_length >= target_length. zero_infinity=True turns a
        # violation into a zero loss instead of an error, so check it up front.
        max_tokens = getattr(config, 'max_bpe_tokens', None)
        if max_tokens is not None and max_tokens > self.num_temporal:
            raise ValueError(
                f"max_bpe_tokens ({max_tokens}) exceeds the CTC time axis "
                f"({self.num_temporal} steps = compressed_frames {self.compressed_frames} "
                f"// tubelet {t_patch}). Those samples would contribute no gradient. "
                f"Either lower max_bpe_tokens to <= {self.num_temporal} or raise "
                f"compressed_frames to >= {max_tokens * t_patch}."
            )

        self._retarget_vivit(frame_h, t_patch, h_patch, w_patch)

        if getattr(config, 'gradient_checkpointing', False):
            # The compressor is trainable and sits before ViViT, so ViViT's
            # activations are kept for backward even when its weights are frozen.
            try:
                self.vivit.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={'use_reentrant': False}
                )
            except TypeError:
                self.vivit.gradient_checkpointing_enable()

        # The compressor and CTC head always stay trainable
        if getattr(config, 'freeze_backbone', True):
            for param in self.vivit.parameters():
                param.requires_grad = False

        hidden_size = self.vivit.config.hidden_size
        self.classifier = nn.Linear(hidden_size, num_classes)
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def _retarget_vivit(self, frame_size, t_patch, h_patch, w_patch):
        """Resize ViViT's position embeddings to our frame count and resolution.

        The checkpoint carries a fixed grid (32 frames at 224px gives 16x14x14 =
        3136 patches plus CLS), so a different frame count or resolution would
        otherwise be a shape mismatch. The pretrained grid is interpolated onto
        ours, and the size attributes HF validates against are updated to match.
        """
        embeddings = self.vivit.embeddings
        if not hasattr(embeddings, 'position_embeddings'):
            raise RuntimeError(
                "This transformers version's VivitEmbeddings has no "
                "`position_embeddings`; cannot interpolate positions."
            )

        vconf = self.vivit.config
        old_t = vconf.num_frames // t_patch
        old_h = vconf.image_size // h_patch
        old_w = vconf.image_size // w_patch

        new_t = self.compressed_frames // t_patch
        new_h = frame_size // h_patch
        new_w = frame_size // w_patch

        pos = embeddings.position_embeddings.data  # (1, 1 + old_t*old_h*old_w, D)
        dim = pos.size(-1)
        expected = old_t * old_h * old_w
        if pos.size(1) != expected + 1:
            raise RuntimeError(
                f"Unexpected position embedding length {pos.size(1)}, expected "
                f"{expected + 1} for a {old_t}x{old_h}x{old_w} grid + CLS"
            )

        if (new_t, new_h, new_w) != (old_t, old_h, old_w):
            cls_pos, grid_pos = pos[:, :1], pos[:, 1:]
            grid_pos = interpolate_position_grid(
                grid_pos, (old_t, old_h, old_w), (new_t, new_h, new_w)
            )
            # The CLS position carries over unchanged
            embeddings.position_embeddings = nn.Parameter(
                torch.cat([cls_pos, grid_pos], dim=1)
            )

        self.position_grid_shape = (new_t, new_h, new_w)

        vconf.num_frames = self.compressed_frames
        vconf.image_size = frame_size
        patch_embeddings = embeddings.patch_embeddings
        patch_embeddings.num_frames = self.compressed_frames
        patch_embeddings.image_size = frame_size
        patch_embeddings.num_patches = new_t * new_h * new_w

    @property
    def output_length(self):
        """Number of CTC time steps emitted, identical for every sample."""
        return self.num_temporal

    def forward(self, x, video_lengths=None):
        # x: (B, C, T_in, H, W)
        x = self.compressor(x, video_lengths)

        # ViViT expects (B, T, C, H, W)
        x = x.permute(0, 2, 1, 3, 4)

        outputs = self.vivit(x)
        sequence_output = outputs.last_hidden_state  # (B, 1 + Tt*S, D)

        # Drop CLS and split the flattened patch grid back into time and space.
        # HF flattens the conv output as (T, H, W) with T slowest, so time is the
        # outer dimension.
        patches = sequence_output[:, 1:, :]
        batch_size, num_patches, dim = patches.shape
        expected = self.num_temporal * self.num_spatial
        if num_patches != expected:
            raise RuntimeError(
                f"ViViT returned {num_patches} patches, expected {expected} "
                f"({self.num_temporal} temporal x {self.num_spatial} spatial)"
            )
        patches = patches.view(batch_size, self.num_temporal, self.num_spatial, dim)

        # Pool away space so CTC runs over time alone
        pooled = patches.mean(dim=2)  # (B, Tt, D)

        logits = self.classifier(pooled)
        log_probs = F.log_softmax(logits.float(), dim=2)

        # Batch stays on dim 0 so nn.DataParallel gathers replicas correctly.
        # The trainer permutes to (T, B, C) for the CTC loss.
        return log_probs

    def unfreeze_backbone(self):
        """Unfreeze the ViViT backbone for fine-tuning."""
        for param in self.vivit.parameters():
            param.requires_grad = True
