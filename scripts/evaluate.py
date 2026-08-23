"""Load a trained checkpoint and score it on a split.

Training only ever reports validation numbers. This is what produces the test-set
results, and what turns a saved checkpoint back into a usable model.

    python scripts/evaluate.py --model vivit --split test
    python scripts/evaluate.py --model vivit --checkpoint path/to/best_model.pth
    python scripts/evaluate.py --model vivit --split test --save-predictions preds.json

By default it reads best_model.pth from the config's checkpoint directory and
scores the full split, ignoring any max_val_batches cap used during training.
"""
import os
import sys
import json
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from src.utils.metrics import calculate_all_metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained checkpoint")
    parser.add_argument('--model', choices=['vivit', 'cnn_bilstm'], default='vivit')
    parser.add_argument('--config', choices=['prod', 'test'], default='prod',
                        help="Which config variant the checkpoint was trained with")
    parser.add_argument('--split', choices=['test', 'val', 'train'], default='test')
    parser.add_argument('--checkpoint', default=None,
                        help="Checkpoint file (defaults to best_model.pth in the "
                             "config's checkpoint directory)")
    parser.add_argument('--save-predictions', default=None,
                        help="Write every target/prediction pair to this JSON file")
    parser.add_argument('--limit', type=int, default=None,
                        help="Only score this many batches (debugging)")
    args = parser.parse_args()

    if args.model == 'vivit':
        if args.config == 'test':
            from configs.test_vivit_ctc_config import config
        else:
            from configs.vivit_ctc_config import config
        from src.training.vivit_trainer import ViViTTrainer as Trainer
    else:
        if args.config == 'test':
            from configs.test_cnn_bilstm_ctc_config import config
        else:
            from configs.cnn_bilstm_ctc_config import config
        from src.training.cnn_bilstm_trainer import CNNBiLSTMTrainer as Trainer

    # Never resume inside an evaluation run; the checkpoint is loaded explicitly
    config.auto_resume = False

    print(f"=== EVALUATION ===")
    print(f"Model: {args.model} ({args.config} config)   split: {args.split}")

    trainer = Trainer(config)

    checkpoint_path = args.checkpoint or os.path.join(
        trainer.checkpoint_dir, 'best_model.pth')
    if not os.path.exists(checkpoint_path):
        print(f"\nNo checkpoint at {checkpoint_path}")
        print("Train first, or pass --checkpoint with the right path.")
        sys.exit(1)

    checkpoint = torch.load(checkpoint_path, map_location=config.device)
    saved_classes = checkpoint.get('num_classes')
    if saved_classes is not None and saved_classes != trainer.num_classes:
        print(f"\nCheckpoint has {saved_classes} output classes but this config "
              f"builds {trainer.num_classes}. The tokenizer or vocabulary changed.")
        sys.exit(1)

    base_model = getattr(trainer.model, 'module', trainer.model)
    base_model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded {checkpoint_path}")
    print(f"  trained to epoch {checkpoint.get('epoch', '?')}, "
          f"val loss {checkpoint.get('val_loss', float('nan')):.4f}")

    loader = {'test': trainer.test_loader,
              'val': trainer.val_loader,
              'train': trainer.train_loader}[args.split]
    print(f"Scoring {len(loader)} batches" + (f" (limited to {args.limit})" if args.limit else ""))

    targets, predictions = trainer.decode_loader(loader, max_batches=args.limit)
    metrics = calculate_all_metrics(targets, predictions)

    empty = sum(1 for p in predictions if not p.strip())
    print(f"\n=== RESULTS on {args.split} ({len(predictions)} samples) ===")
    print(f"  WER: {metrics['wer']:.4f}")
    print(f"  CER: {metrics['cer']:.4f}")
    print(f"  Exact match: {metrics['exact_match_accuracy']:.4f}")
    print(f"  Token accuracy: {metrics['token_accuracy']:.4f}")
    print(f"  Empty predictions: {empty} ({100 * empty / max(1, len(predictions)):.1f}%)")
    if empty == len(predictions):
        print("  Every prediction is empty, which is what CTC collapsing to all-blank"
              " looks like. WER near 1.0 here means the model is not emitting tokens.")

    print("\nExamples:")
    for target, prediction in list(zip(targets, predictions))[:5]:
        print(f"  target: {target[:90]}")
        print(f"  pred  : {prediction[:90]}")

    if args.save_predictions:
        os.makedirs(os.path.dirname(os.path.abspath(args.save_predictions)) or '.',
                    exist_ok=True)
        with open(args.save_predictions, 'w', encoding='utf-8') as f:
            json.dump({
                'model': args.model,
                'config': args.config,
                'split': args.split,
                'checkpoint': checkpoint_path,
                'epoch': checkpoint.get('epoch'),
                'metrics': metrics,
                'pairs': [{'target': t, 'prediction': p}
                          for t, p in zip(targets, predictions)],
            }, f, ensure_ascii=False, indent=2)
        print(f"\nWrote predictions to {args.save_predictions}")


if __name__ == "__main__":
    main()
