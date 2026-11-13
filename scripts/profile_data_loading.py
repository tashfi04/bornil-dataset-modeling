import os
import sys
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.base_config import config
from src.data_loader import get_data_loaders

def profile_data_loading():
    """Profile each step of the data loading pipeline"""
    train_loader, _, _ = get_data_loaders(config)

    print("=== DATA LOADING PROFILING ===")
    print(f"Batch size: {config.batch_size}")
    print(f"Num workers: {config.num_workers}")

    # Profile a few batches
    total_video_time = 0
    total_collate_time = 0
    batch_count = 0

    start_time = time.time()

    for i, batch in enumerate(train_loader):
        if i >= 10:  # Profile 10 batches
            break

        batch_start = time.time()

        # Time video loading in dataset
        dataset_time = batch_start - start_time

        # Time collate function
        collate_start = time.time()
        _ = batch  # Just accessing the batch
        collate_time = time.time() - collate_start

        total_video_time += dataset_time
        total_collate_time += collate_time
        batch_count += 1

        print(f"Batch {i}: Dataset={dataset_time:.3f}s, Collate={collate_time:.3f}s")

        start_time = time.time()

    if batch_count > 0:
        print(f"\n=== AVERAGES ===")
        print(f"Dataset loading: {total_video_time/batch_count:.3f}s per batch")
        print(f"Collate function: {total_collate_time/batch_count:.3f}s per batch")
        print(f"Total per batch: {(total_video_time + total_collate_time)/batch_count:.3f}s")
        print(f"Estimated batches per epoch: {len(train_loader)}")
        print(f"Estimated epoch time: {(total_video_time + total_collate_time)/batch_count * len(train_loader)/60:.1f} minutes")

if __name__ == "__main__":
    profile_data_loading()
