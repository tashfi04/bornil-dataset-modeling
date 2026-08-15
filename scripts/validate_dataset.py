import pandas as pd
import json
import os
import sys
import cv2

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.base_config import config
from src.utils.text_utils import text_to_int

def validate_dataset():
    """Simple validation: Check for CTC requirement violations and missing files"""

    # Load data
    df = pd.read_csv(config.csv_path)

    # Load vocabulary for text length calculation
    with open(config.vocab_path, 'r', encoding='utf-8') as f:
        vocab = json.load(f)
    char_to_id = vocab['char_to_id']

    # Load splits
    with open(config.train_val_test_split_path, 'r', encoding='utf-8') as f:
        splits = json.load(f)

    print("=== DATASET VALIDATION ===")
    print(f"Total samples in splits: {len(splits)}")

    # Track issues
    issues = {
        'missing_files': [],
        'ctc_violations': [],  # video_length < text_length
        'zero_length_videos': [],
        'zero_length_text': []
    }

    # Check each sample
    for i, (recording_id, split) in(splits.items()):
        if i % 1000 == 0:
            print(f"Processed {i}/{len(splits)} samples...")

        # Find the row in dataframe
        sample_df = df[df['recording'] == recording_id]
        if len(sample_df) == 0:
            issues['missing_files'].append(recording_id)
            continue

        row = sample_df.iloc[0]
        text = row['text']

        # Check text length
        text_seq = text_to_int(text, char_to_id)
        text_length = len(text_seq)

        if text_length == 0:
            issues['zero_length_text'].append(recording_id)
            continue

        # Check video file existence and length
        video_path = os.path.join(config.chunk_base_path, row['chunk_path'], recording_id)
        if not os.path.exists(video_path):
            issues['missing_files'].append(recording_id)
            continue

        # Get video length (number of frames)
        try:
            cap = cv2.VideoCapture(video_path)
            video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()

            if video_length == 0:
                issues['zero_length_videos'].append(recording_id)
                continue

            # Check CTC requirement
            if video_length < text_length:
                issues['ctc_violations'].append({
                    'recording': recording_id,
                    'video_length': video_length,
                    'text_length': text_length,
                    'text': text
                })

        except Exception as e:
            issues['missing_files'].append(recording_id)  # Treat as missing if can't read

    # Print summary
    print("\n=== VALIDATION RESULTS ===")
    total_issues = sum(len(issue_list) for issue_list in issues.values())
    print(f"Total issues found: {total_issues}")
    print(f"Clean samples: {len(splits) - total_issues} ({((len(splits) - total_issues)/len(splits))*100:.1f}%)")

    print("\nIssue breakdown:")
    for issue_type, samples in issues.items():
        print(f"  {issue_type}: {len(samples)}")

        # Show first few examples for each issue type
        if samples and issue_type == 'ctc_violations':
            print("    Examples (first 5):")
            for i, sample in enumerate(samples[:5]):
                print(f"      {sample['recording']}: video={sample['video_length']} frames, text={sample['text_length']} chars")
                print(f"        Text: '{sample['text']}'")
        elif samples and issue_type != 'ctc_violations':
            print(f"    Examples (first 5): {samples[:5]}")

    # Recommendations
    if total_issues > 0:
        print(f"\n RECOMMENDATIONS:")
        if issues['missing_files']:
            print("  - Check file paths and fix missing videos")
        if issues['ctc_violations']:
            print("  - Review videos with insufficient frames for their text length")
            print("  - Consider: shorter text labels or video segmentation")
        if issues['zero_length_videos']:
            print("  - Remove or fix zero-length videos")
        if issues['zero_length_text']:
            print("  - Remove samples with empty text")

        print(f"\nRun this to see all problematic files:")
        print(f"  python scripts/validate_dataset.py > validation_report.txt")
    else:
        print("Dataset is clean! No issues found.")

    return issues

if __name__ == "__main__":
    validate_dataset()