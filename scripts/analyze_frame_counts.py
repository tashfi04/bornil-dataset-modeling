import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.base_config import config
from src.data_loader import get_data_loaders

def analyze_actual_frame_counts():
    """Analyze actual frame counts from successfully loaded videos"""
    train_loader, val_loader, test_loader = get_data_loaders(config)

    all_frame_counts = []

    print("=== ACTUAL FRAME COUNT ANALYSIS ===")

    def collect_frames(loader, name):
        frame_counts = []
        total_batches = 0

        for batch in loader:
            total_batches += 1
            for i, path in enumerate(batch['video_paths']):
                if path != 'failed_to_load':  # Only count successful loads
                    frame_counts.append(batch['video_lengths'][i].item())

            # Sample enough to get good statistics
            if total_batches > 50 and len(frame_counts) > 1000:
                break

        print(f"{name}: {len(frame_counts)} successful videos")
        return frame_counts

    train_frames = collect_frames(train_loader, "Train")
    val_frames = collect_frames(val_loader, "Validation") 
    test_frames = collect_frames(test_loader, "Test")

    all_frame_counts = train_frames + val_frames + test_frames

    if not all_frame_counts:
        print("❌ No videos loaded successfully! Check your paths.")
        return

    print(f"\n=== FRAME COUNT STATISTICS ===")
    print(f"Total videos analyzed: {len(all_frame_counts)}")
    print(f"Min frames: {min(all_frame_counts)}")
    print(f"Max frames: {max(all_frame_counts)}")
    print(f"Mean frames: {np.mean(all_frame_counts):.1f}")
    print(f"Median frames: {np.median(all_frame_counts):.1f}")

    # Percentile analysis
    percentiles = [50, 75, 90, 95, 99]
    print(f"\nPercentiles:")
    for p in percentiles:
        value = np.percentile(all_frame_counts, p)
        print(f"  {p}th: {value:.0f} frames")

    # Distribution analysis
    print(f"\nDistribution:")
    ranges = [(0, 100), (101, 300), (301, 1000), (1001, 2000), (2001, 5000), (5001, float('inf'))]
    for min_f, max_f in ranges:
        if max_f == float('inf'):
            count = sum(1 for f in all_frame_counts if f >= min_f)
        else:
            count = sum(1 for f in all_frame_counts if min_f <= f <= max_f)
        percentage = count / len(all_frame_counts) * 100
        print(f"  {min_f}-{max_f}: {count} videos ({percentage:.1f}%)")

    # Recommend max_frames based on 95th percentile
    p95 = np.percentile(all_frame_counts, 95)
    print(f"\n🎯 RECOMMENDATION: Set max_frames = {int(p95)} (95th percentile)")
    print(f"   This would affect only {100-95}% of videos")

if __name__ == "__main__":
    analyze_actual_frame_counts()
