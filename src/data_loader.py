import os
import json
import logging
from functools import partial

import cv2
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader

from src.utils.text_utils import text_to_int, load_bpe_tokenizer, text_to_bpe_ids
from src.utils.frame_cache import cache_path, read_cached_frames
from src.utils.ctc_limits import ctc_time_steps, target_token_limit

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Shared with scripts/preprocess_cache_frames.py so cached frames match what
# the dataset would produce on the fly.
def uniform_sample_frames(frames, target_frames):
    indices = np.linspace(0, len(frames) - 1, target_frames, dtype=int)
    return [frames[i] for i in indices]


def strategic_sample_frames(frames, target_frames, sampling_segments=3):
    """Strategic sampling focusing on key segments"""
    frames_per_segment = target_frames // sampling_segments
    indices = []

    for i in range(sampling_segments):
        # Sample from different segments of the video
        start = (i * len(frames)) // sampling_segments
        end = ((i + 1) * len(frames)) // sampling_segments

        # Ensure we don't sample beyond available frames
        segment_frames = min(frames_per_segment, end - start)
        if segment_frames > 0:
            segment_indices = np.linspace(start, end - 1, segment_frames, dtype=int)
            indices.extend(segment_indices)

    # If we have leftover frames due to integer division, sample from middle
    remaining_frames = target_frames - len(indices)
    if remaining_frames > 0:
        middle_start = len(frames) // 3
        middle_end = 2 * len(frames) // 3
        extra_indices = np.linspace(middle_start, middle_end - 1, remaining_frames, dtype=int)
        indices.extend(extra_indices)

    # Sort indices to maintain temporal order
    indices.sort()
    return [frames[i] for i in indices]


def sample_frames(frames, target_frames, strategy='uniform', sampling_segments=3):
    """Reduce `frames` to `target_frames` using the configured strategy.
    Videos already at or below the target are returned untouched."""
    if len(frames) <= target_frames:
        return frames
    if strategy == 'strategic':
        return strategic_sample_frames(frames, target_frames, sampling_segments)
    return uniform_sample_frames(frames, target_frames)

class BdSLDataset(Dataset):
    def __init__(self, df, config, mode='train'):
        """
        Args:
            df: DataFrame with video metadata
            config: Configuration object (from base_config or model-specific config)
            mode: 'train', 'val', or 'test'
        """
        self.df = df.reset_index(drop=True)
        self.mode = mode
        self.config = config    # Store the config object

        # Tokenizer setup
        self.tokenization_type = getattr(config, 'tokenization_type', 'character')
        if self.tokenization_type == 'bpe':
            self.tokenizer = load_bpe_tokenizer(self.config.bpe_tokenizer_path)
            self.char_to_id = None  # not used
        else:  # Default character level tokenization
            with open(self.config.vocab_path, 'r', encoding='utf-8') as f:
                vocab = json.load(f)
            self.char_to_id = vocab['char_to_id']
            self.tokenizer = None   # not used

        # In strict mode a sample that fails to load aborts the run instead of
        # being dropped, so training never silently proceeds on partial data
        self.strict = bool(getattr(config, 'strict_data', True))

        # Recordings cleared by scripts/validate_dataset.py. When present this is
        # the authoritative list and no further filtering happens at runtime.
        self.approved = self._load_approved_samples()

        # When set, frames come from the cache built by
        # scripts/preprocess_cache_frames.py instead of being decoded per epoch
        self.cache_root = getattr(config, 'cached_frames_path', None) or None
        if self.cache_root is not None:
            logger.info(f"Reading frames from cache: {self.cache_root}")

        self.valid_indices = self._build_valid_indices()

        logger.info(f"Initialized {mode} dataset with {len(self.valid_indices)} valid samples (out of {len(self.df)})")
        logger.info(f"Using config: {self.config.model_type if hasattr(self.config, 'model_type') else 'base'}")
        logger.info(f"Tokenization type: {self.tokenization_type}")

    def _load_approved_samples(self):
        """Read the valid-sample list written by scripts/validate_dataset.py."""
        path = getattr(self.config, 'valid_samples_path', None)
        if not path:
            return None
        if not os.path.exists(path):
            # Falling back still filters missing files and over-long targets, but
            # cannot catch a video that fails to decode
            logger.warning(
                f"No pre-validation list at {path}; filtering at load time instead. "
                f"Run scripts/validate_dataset.py to check every video up front."
            )
            return None

        with open(path, 'r', encoding='utf-8') as f:
            payload = json.load(f)

        # These decide which samples are valid at all, so a mismatch means the
        # list admits samples this model's CTC axis cannot align (or excludes
        # samples it could have used).
        critical = {
            'tokenization': getattr(self.config, 'tokenization_type', 'character'),
            'ctc_steps': ctc_time_steps(self.config),
            'token_limit': target_token_limit(self.config),
        }
        mismatched = {
            key: (payload[key], current)
            for key, current in critical.items()
            if key in payload and payload[key] != current
        }
        if mismatched:
            details = ', '.join(
                f"{key}={recorded} (config has {current})"
                for key, (recorded, current) in mismatched.items()
            )
            message = (f"{path} was generated with {details}. Re-run "
                       f"scripts/validate_dataset.py for this config.")
            if self.strict:
                raise RuntimeError(message)
            logger.warning(message)

        # num_frames does not change validity, only whether short videos get
        # zero-padded, so it is worth reporting but not worth refusing to run
        recorded_frames = payload.get('num_frames')
        if recorded_frames is not None and recorded_frames != self.config.num_frames:
            logger.info(
                f"{path} was generated with num_frames={recorded_frames}; this run "
                f"uses {self.config.num_frames}. The sample list is still valid."
            )

        if payload.get('quick'):
            logger.warning(f"{path} was generated with --quick, so undecodable "
                           f"videos may not have been caught")

        return set(payload['valid'])

    def _source_path(self, row):
        """Cached clip if a cache is configured, otherwise the raw video."""
        if self.cache_root is not None:
            return cache_path(self.cache_root, row['chunk_path'], row['recording'])
        return os.path.join(self.config.chunk_base_path, row['chunk_path'], row['recording'])

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        # Map to original dataframe index
        real_idx = self.valid_indices[idx]
        row = self.df.iloc[real_idx]

        text_label = row['text']

        full_video_path = self._source_path(row)


        # Pre-validation should have removed anything unreadable, so a failure
        # here means the dataset changed or a file is corrupt
        max_attempts = 3
        video = None
        last_error = None
        for attempt in range(max_attempts):
            try:
                video = self.load_video_frames(full_video_path)
                break
            except Exception as e:
                last_error = e
                logger.warning(f"Error loading video {full_video_path} (attempt {attempt + 1}/{max_attempts}): {e}")

        if video is None:
            message = (f"Failed to load {full_video_path} after {max_attempts} attempts: "
                       f"{last_error}")
            if self.strict:
                raise RuntimeError(
                    message + ". Re-run scripts/validate_dataset.py to refresh the "
                    "valid-sample list, or set strict_data=False to skip such samples."
                )
            logger.error(message)
            return None

        # Convert text to integer sequence
        if self.tokenization_type == 'bpe':
            text_seq = text_to_bpe_ids(text_label, self.tokenizer)
        else:  # Default character level tokenization
            text_seq = text_to_int(text_label, self.char_to_id)

        return {
            'video': torch.FloatTensor(video),
            'text': text_label,
            'text_seq': torch.LongTensor(text_seq),
            'video_path': full_video_path,
            'loaded_successfully': True  # Flag for successful loading video
        }

    def load_video_frames(self, video_path):
        """Load and preprocess video frames using config parameters"""
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        target_frames = getattr(self.config, 'num_frames', 160)
        sampling_strategy = getattr(self.config, 'sampling_strategy', 'uniform')

        if self.cache_root is not None:
            # Cached clips are already sampled and resized
            frames = read_cached_frames(video_path)
        else:
            cap = cv2.VideoCapture(video_path)
            frames = []

            try:
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frame = cv2.resize(frame, (self.config.frame_size[1], self.config.frame_size[0]))
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(frame)
            finally:
                cap.release()

            if len(frames) == 0:
                raise ValueError(f"No frames loaded from {video_path}")

            frames = sample_frames(
                frames, target_frames, sampling_strategy,
                getattr(self.config, 'sampling_segments', 3)
            )

        # Padding for shorter videos is handled in collate_fn
        # Normalize and reshape
        frames = np.array(frames, dtype=np.float32) / 255.0
        frames = np.transpose(frames, (3, 0, 1, 2))  # (C, T, H, W)

        return frames

    def _uniform_sample_frames(self, frames, target_frames):
        return uniform_sample_frames(frames, target_frames)

    def _strategic_sample_frames(self, frames, target_frames):
        return strategic_sample_frames(
            frames, target_frames, getattr(self.config, 'sampling_segments', 3)
        )

    def _build_valid_indices(self):
        """Indices of the samples this split will actually train on."""
        if self.approved is not None:
            valid_indices = [
                idx for idx in range(len(self.df))
                if self.df.iloc[idx]['recording'] in self.approved
            ]
            logger.info(
                f"{self.mode}: {len(valid_indices)} / {len(self.df)} samples "
                f"approved by the pre-validation list"
            )
            return valid_indices

        valid_indices = []
        dropped_bpe = 0
        missing_videos = 0

        for idx in range(len(self.df)):
            row = self.df.iloc[idx]
            full_video_path = self._source_path(row)

            if not os.path.exists(full_video_path):
                missing_videos += 1
                logger.info(f"Sample {idx} (recording: {row['recording']}) source file missing: {full_video_path}")
                continue

            if self.tokenization_type == 'bpe':
                try:
                    bpe_ids = text_to_bpe_ids(row['text'], self.tokenizer)
                    if len(bpe_ids) > self.config.max_bpe_tokens:
                        dropped_bpe += 1
                        continue
                except Exception:
                    dropped_bpe += 1
                    continue

            valid_indices.append(idx)

        logger.info(f"Valid samples: {len(valid_indices)} / {len(self.df)}")
        logger.info(f"Dropped (missing files): {missing_videos}")
        logger.info(f"Dropped (BPE too long): {dropped_bpe}")

        return valid_indices

def get_data_loaders(config):
    """
    Create train, validation, and test data loaders

    Args:
        config: Configuration object containing all necessary parameters
    """
    try:
        df = pd.read_csv(config.csv_path)
        logger.info(f"Loaded dataset with {len(df)} total samples")
    except Exception as e:
        logger.error(f"Error loading CSV: {e}")
        raise

    # Load fixed splits
    split_file = config.train_val_test_split_path
    if not os.path.exists(split_file):
        raise FileNotFoundError(
            f"Split file not found: {split_file}. "
            f"Please run scripts/generate_splits.py first."
        )

    with open(split_file, 'r', encoding='utf-8') as f:
        split_mapping = json.load(f)

    # Split the data based on saved mapping
    train_df = df[df['recording'].isin([k for k, v in split_mapping.items() if v == 'train'])]
    val_df = df[df['recording'].isin([k for k, v in split_mapping.items() if v == 'val'])]
    test_df = df[df['recording'].isin([k for k, v in split_mapping.items() if v == 'test'])]

    logger.info(f"Using fixed splits - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    batch_collate = partial(collate_fn, strict=bool(getattr(config, 'strict_data', True)))

    # Create datasets
    train_dataset = BdSLDataset(train_df, config, 'train')
    val_dataset = BdSLDataset(val_df, config, 'val')
    test_dataset = BdSLDataset(test_df, config, 'test')

    # Use config parameters for DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=batch_collate,
        pin_memory=True,
        persistent_workers=True,    # Keep workers alive between epochs
        prefetch_factor=2           # Prefetch batches
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=batch_collate,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=batch_collate,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader

def collate_fn(batch, strict=True):
    """Collate variable-length videos and text sequences into a padded batch."""
    dropped = sum(1 for b in batch if b is None)
    if dropped and strict:
        raise RuntimeError(
            f"{dropped} of {len(batch)} samples failed to load. Re-run "
            "scripts/validate_dataset.py, or set strict_data=False to train on "
            "the remainder."
        )

    batch = [b for b in batch if b is not None]
    if dropped:
        logger.error(f"Dropped {dropped} unreadable sample(s) from this batch")

    if len(batch) == 0:
        raise RuntimeError("Empty batch received in collate_fn!!!")

    # Sort batch by video length (descending) for packed sequences
    batch.sort(key=lambda x: x['video'].size(1), reverse=True)

    videos = [item['video'] for item in batch]
    text_seqs = [item['text_seq'] for item in batch] # list of 1D LongTensors (variable len)
    text_labels = [item['text'] for item in batch]
    video_paths = [item['video_path'] for item in batch]

    # Get actual video lengths (number of frames)
    video_lengths = torch.LongTensor([video.size(1) for video in videos])

    # Pad videos to maximum length in this batch
    padded_videos = torch.nn.utils.rnn.pad_sequence(
        [video.permute(1, 0, 2, 3) for video in videos],  # (T, C, H, W) for padding
        batch_first=True
    ).permute(0, 2, 1, 3, 4)  # Back to (B, C, T, H, W)

    # Get lengths for CTC loss
    text_lengths = torch.LongTensor([len(seq) for seq in text_seqs])

    # Create 1D concatenated text targets (required by nn.CTCLoss)
    if len(text_seqs) > 0:
        text_targets = torch.cat(text_seqs).long()  # 1D LongTensor of all targets
    else:
        text_targets = torch.LongTensor([]).long()

    padded_text_seqs = torch.nn.utils.rnn.pad_sequence(text_seqs, batch_first=True)

    return {
        'videos': padded_videos,
        'text_targets': text_targets,   # 1D targets for CTCLoss
        'text_seqs': padded_text_seqs,
        'text_labels': text_labels,
        'video_paths': video_paths,
        'video_lengths': video_lengths, # Actual lengths before padding
        'text_lengths': text_lengths
    }

def check_dataset_health(config, sample_size=5):
    """
    Dry-run to ensure the dataloader, transformations,
    and collate_fn are working correctly without crashing
    """
    from src.data_loader import get_data_loaders

    try:
        train_loader, val_loader, test_loader = get_data_loaders(config)
    except Exception as e:
        print(f"Dataset health check failed during DataLoader initialization: {e}")
        return False

    def check_loader(loader, name):
        total_samples = 0
        frame_counts = []
        batches_seen = 0

        for i, batch in enumerate(loader):
            if i >= sample_size:  # Check only sample_size batches
                break

            batches_seen = i + 1
            batch_size = len(batch['video_paths'])
            total_samples += batch_size
            for length in batch['video_lengths']:
                frame_counts.append(length.item())

        if total_samples == 0:
            print(f"Dataset health check failed: {name} dataLoader returned 0 samples")
            return False

        print(f"{name}: successfully loaded {total_samples} samples across {batches_seen} batches")
        if frame_counts:
            print(f"Frame count stats: min={min(frame_counts)}, max={max(frame_counts)}, avg={sum(frame_counts)/len(frame_counts):.1f}")

        return True

    print("=== DATASET HEALTH CHECK DRY RUN ===")
    train_ok = check_loader(train_loader, "Train")
    val_ok = check_loader(val_loader, "Validation")
    test_ok = check_loader(test_loader, "Test")

    # Return True only if all loaders successfully yield data
    return train_ok and val_ok and test_ok
