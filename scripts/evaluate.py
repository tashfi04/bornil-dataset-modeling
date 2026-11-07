"""Load a trained checkpoint and score it on a split.

Training only ever reports validation numbers. This is what produces the test-set
results, and what turns a saved checkpoint back into a usable model.

    python scripts/evaluate.py --model vivit --split test
    python scripts/evaluate.py --model vivit --checkpoint path/to/best_model.pth
    python scripts/evaluate.py --model vivit --split test --save-predictions preds.json

By default it reads best_model.pth from the config's checkpoint directory,
greedy-decodes and scores the full split, ignoring any max_val_batches cap used
during training. Beam search, optionally with a language model from
scripts/train_ngram_lm.py, has its settings chosen on val and then applied to
test unchanged:

    python scripts/evaluate.py --model vivit --split val --decoder beam
    python scripts/evaluate.py --model vivit --split val --lm path/to/bpe_4gram.arpa --tune-lm
    python scripts/evaluate.py --model vivit --split test --lm path/to/bpe_4gram.arpa \\
        --lm-weight 0.5 --insertion-bonus 1.0

Every run also reports token detection (src/utils/token_detection.py): whether
the model's output can tell which tokens a clip contains, which shows what a
model has learned even when it decodes to almost nothing. It also rescores the
predictions against other clips' sentences, as a floor that needs no video.

Results sit with the model's other outputs, e.g. for the overfit config:

    outputs/vivit_ctc_overfit/checkpoints/                   (training)
    outputs/vivit_ctc_overfit/training.log                   (training)
    outputs/vivit_ctc_overfit/evaluation/<split>_<checkpoint name>[_beam[_lm]].json
    outputs/vivit_ctc_overfit/evaluate.log
"""
import os
import sys
import json
import random
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from src.utils.metrics import calculate_all_metrics, compute_cer, compute_wer
from src.utils.text_utils import (label_fingerprint_mismatch, output_units,
                                  text_to_bpe_ids, text_to_int)
from src.utils.environment import checkpoint_environment_mismatch
from src.utils.ctc_decoding import beam_search_all
from src.utils.ngram_lm import read_arpa, metadata_path
from src.utils.token_detection import token_detection

# Non-blank ids beam search considers at each time step
TOKEN_CANDIDATES = 16

# The grid --tune-lm searches on val
LM_WEIGHTS = (0.1, 0.2, 0.3, 0.5, 0.7, 1.0)
INSERTION_BONUSES = (0.0, 0.5, 1.0, 2.0, 3.0)

# A token is scored for detection only if at least this many clips contain it
# and this many do not
DETECTION_MIN_CLIPS = 10


def load_language_model(path, trainer, config):
    """Read an ARPA file and confirm it was built for this model's output units."""
    def stop(message):
        print(f"\n{message}")
        sys.exit(1)

    if not os.path.exists(path):
        stop(f"No language model at {path}. Build one with scripts/train_ngram_lm.py.")
    record_path = metadata_path(path)
    if not os.path.exists(record_path):
        stop(f"{record_path} is missing, so {path} cannot be confirmed to match this "
             f"model's tokenizer. Rebuild it with scripts/train_ngram_lm.py.")
    with open(record_path, 'r', encoding='utf-8') as f:
        record = json.load(f)
    if record.get('label_fingerprint') != trainer.label_fingerprint:
        stop(f"{path} was built for tokenizer {record.get('label_fingerprint')}, but this "
             f"model uses {trainer.label_fingerprint}. Rebuild it with "
             f"scripts/train_ngram_lm.py for this config.")

    lm = read_arpa(path)
    units = output_units(config)
    missing = [unit for unit in units[1:] if unit not in lm.vocabulary]
    if missing:
        stop(f"{path} has no probability for {len(missing)} of this model's units "
             f"(first: {missing[:5]}). Rebuild it with scripts/train_ngram_lm.py.")
    return lm, units, record


def clip_steps(blank, ids, logprobs):
    """One clip's collected outputs in the form prefix_beam_search takes."""
    return [(b, list(zip(i, l))) for b, i, l in zip(blank.tolist(), ids.tolist(), logprobs.tolist())]


def reference_ids(trainer, texts):
    if trainer.tokenization_type == 'bpe':
        return [text_to_bpe_ids(text, trainer.tokenizer) for text in texts]
    return [text_to_int(text, trainer.vocab['char_to_id']) for text in texts]


def shuffled_references(targets, seed=0):
    """The references reassigned along a random cycle, so no clip keeps its own."""
    order = list(range(len(targets)))
    random.Random(seed).shuffle(order)
    shuffled = list(targets)
    for k, clip in enumerate(order):
        shuffled[clip] = targets[order[(k + 1) % len(order)]]
    return shuffled


def tune_lm(trainer, steps, targets, search, workers):
    """Try every LM weight and insertion bonus in the grid; return them by val WER.

    Not by CER: when the model is mostly guessing, longer guesses share more
    common letters with the reference by chance, so the largest insertion bonus
    always wins on CER without recognising anything more.
    """
    tried = []
    print(f"\nTuning on {len(steps)} val clips: {len(LM_WEIGHTS)} LM weights x "
          f"{len(INSERTION_BONUSES)} insertion bonuses, by WER")
    for lm_weight in LM_WEIGHTS:
        for insertion_bonus in INSERTION_BONUSES:
            ids = beam_search_all(steps, workers, lm_weight=lm_weight,
                                  insertion_bonus=insertion_bonus, **search)
            predictions = [trainer.ids_to_text(x) for x in ids]
            tried.append({'lm_weight': lm_weight, 'insertion_bonus': insertion_bonus,
                          'cer': compute_cer(targets, predictions),
                          'wer': compute_wer(targets, predictions)})
            print(f"  weight {lm_weight:<4} bonus {insertion_bonus:<4} "
                  f"WER {tried[-1]['wer']:.4f}  CER {tried[-1]['cer']:.4f}")
    return sorted(tried, key=lambda t: (t['wer'], t['cer']))


def print_metrics(columns):
    """Print one column of metrics per decoder."""
    names = list(columns)
    print("  " + " " * 16 + "".join(f"{name:>12}" for name in names))
    rows = [('WER', 'wer', '.4f'), ('CER', 'cer', '.4f')]
    rows += [(f'BLEU-{n}', f'bleu{n}', '.2f') for n in range(1, 5)]
    rows += [('ROUGE-L', 'rouge_l', '.2f'), ('chrF', 'chrf', '.2f'),
             ('Exact match', 'exact_match_accuracy', '.4f'),
             ('Token accuracy', 'token_accuracy', '.4f')]
    for label, key, spec in rows:
        print(f"  {label:<16}" + "".join(f"{columns[name][key]:>12{spec}}" for name in names))


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained checkpoint")
    parser.add_argument('--model', choices=['vivit', 'cnn_bilstm'], default='vivit')
    parser.add_argument('--config', choices=['prod', 'test', 'overfit', 'subset'],
                        default='prod',
                        help="Which config variant the checkpoint was trained with")
    parser.add_argument('--csv', default=None,
                        help="Dataset CSV (defaults to the config's csv_path)")
    parser.add_argument('--no-cache', action='store_true',
                        help="Decode the videos instead of reading the frame cache")
    parser.add_argument('--frozen-backbone', action='store_true',
                        help="With --config subset: evaluate the frozen-backbone run")
    parser.add_argument('--split', choices=['test', 'val', 'train'], default='test')
    parser.add_argument('--checkpoint', default=None,
                        help="Checkpoint file (defaults to best_model.pth in the "
                             "config's checkpoint directory)")
    parser.add_argument('--save-predictions', default=None,
                        help="Results file (defaults to <split>_<checkpoint name>.json "
                             "in the model's evaluation directory)")
    parser.add_argument('--limit', type=int, default=None,
                        help="Only score this many batches (debugging)")
    parser.add_argument('--decoder', choices=['greedy', 'beam'], default=None,
                        help="greedy (the default) or beam; --lm implies beam")
    parser.add_argument('--beam-size', type=int, default=16)
    parser.add_argument('--lm', default=None,
                        help="ARPA language model from scripts/train_ngram_lm.py")
    parser.add_argument('--lm-weight', type=float, default=None,
                        help="Weight of the LM score; use the value --tune-lm chose on val")
    parser.add_argument('--insertion-bonus', type=float, default=None,
                        help="Score added per emitted unit; use the value --tune-lm chose on val")
    parser.add_argument('--tune-lm', action='store_true',
                        help="Choose --lm-weight and --insertion-bonus on val by WER")
    parser.add_argument('--workers', type=int, default=min(4, os.cpu_count() or 1),
                        help="Processes for beam search")
    args = parser.parse_args()

    if args.frozen_backbone and not (args.model == 'vivit' and args.config == 'subset'):
        parser.error("--frozen-backbone only applies to --model vivit --config subset")
    if args.decoder == 'greedy' and (args.lm or args.tune_lm):
        parser.error("--lm and --tune-lm need beam search, not --decoder greedy")
    decoder = 'beam' if (args.decoder == 'beam' or args.lm or args.tune_lm) else 'greedy'
    if args.tune_lm and (not args.lm or args.split != 'val'):
        parser.error("--tune-lm needs --lm and --split val, so test stays unseen until scored")
    if args.lm and not args.tune_lm and (args.lm_weight is None or args.insertion_bonus is None):
        parser.error("--lm needs --lm-weight and --insertion-bonus; choose them on val "
                     "with --tune-lm first")
    if not args.lm and (args.lm_weight is not None or args.insertion_bonus is not None):
        parser.error("--lm-weight and --insertion-bonus only apply with --lm")
    if args.tune_lm and (args.lm_weight is not None or args.insertion_bonus is not None):
        parser.error("--tune-lm chooses --lm-weight and --insertion-bonus itself")

    if args.model == 'vivit':
        if args.config == 'test':
            from configs.test_vivit_ctc_config import config
        elif args.config == 'overfit':
            from configs.overfit_vivit_ctc_config import config
        elif args.config == 'subset':
            from configs.subset_vivit_ctc_config import config, use_frozen_backbone
            if args.frozen_backbone:
                use_frozen_backbone(config)
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
    # Clips are scored as recorded, including with --split train
    config.augment = False
    if args.csv:
        config.csv_path = args.csv
    if args.no_cache:
        config.cached_frames_path = None

    print(f"=== EVALUATION ===")
    print(f"Model: {args.model} ({args.config} config)   split: {args.split}   decoder: {decoder}")

    trainer = Trainer(config, log_filename='evaluate.log')

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
    mismatch = label_fingerprint_mismatch(checkpoint, trainer.label_fingerprint, checkpoint_path)
    if mismatch:
        print(f"\n{mismatch} Its scores would be meaningless.")
        sys.exit(1)
    mismatch = checkpoint_environment_mismatch(checkpoint, trainer.environment, checkpoint_path)
    if mismatch:
        print(f"\n{mismatch} Evaluate it in the environment it was trained in.")
        sys.exit(1)

    lm = units = lm_record = None
    if args.lm:
        lm, units, lm_record = load_language_model(args.lm, trainer, config)
        print(f"Language model: {args.lm} ({lm.order}-gram over {lm_record['units']} units, "
              f"{lm_record['sentences']} sentences)")

    base_model = getattr(trainer.model, 'module', trainer.model)
    base_model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded {checkpoint_path}")
    print(f"  trained to epoch {checkpoint.get('epoch', '?')}, "
          f"val loss {checkpoint.get('val_loss', float('nan')):.4f}")
    if checkpoint.get('selection_score') is not None and checkpoint.get('selection_metric') != 'val_loss':
        print(f"  val {checkpoint['selection_metric']} {checkpoint['selection_score']:.4f}")

    loader = {'test': trainer.test_loader,
              'val': trainer.val_loader,
              'train': trainer.train_loader}[args.split]
    print(f"Scoring {len(loader)} batches" + (f" (limited to {args.limit})" if args.limit else ""))

    collected = trainer.collect_ctc_outputs(loader, max_batches=args.limit, top_k=TOKEN_CANDIDATES)
    targets = collected['targets']
    greedy = collected['greedy']
    predictions = greedy
    settings = {'type': 'greedy'}
    tuning = None

    if decoder == 'beam':
        steps = [clip_steps(*clip) for clip in
                 zip(collected['blank'], collected['ids'], collected['logprobs'])]
        search = {'beam_size': args.beam_size, 'lm': lm, 'units': units}
        if args.tune_lm:
            tuning = tune_lm(trainer, steps, targets, search, args.workers)
            lm_weight, insertion_bonus = tuning[0]['lm_weight'], tuning[0]['insertion_bonus']
            print(f"Best on val: --lm-weight {lm_weight} --insertion-bonus {insertion_bonus} "
                  f"(WER {tuning[0]['wer']:.4f}). Score test with these values, the same "
                  f"--lm and your other options, and without --tune-lm.")
        else:
            lm_weight = args.lm_weight if args.lm else 0.0
            insertion_bonus = args.insertion_bonus if args.lm else 0.0
        ids = beam_search_all(steps, args.workers, lm_weight=lm_weight,
                              insertion_bonus=insertion_bonus, **search)
        predictions = [trainer.ids_to_text(x) for x in ids]
        settings = {'type': 'beam', 'beam_size': args.beam_size,
                    'token_candidates': TOKEN_CANDIDATES, 'lm': args.lm,
                    'lm_order': lm.order if lm else None,
                    'lm_weight': lm_weight, 'insertion_bonus': insertion_bonus}

    metrics = calculate_all_metrics(targets, predictions)
    greedy_metrics = metrics if decoder == 'greedy' else calculate_all_metrics(targets, greedy)

    empty = sum(1 for p in predictions if not p.strip())
    print(f"\n=== RESULTS on {args.split} ({len(predictions)} samples) ===")
    if decoder == 'greedy':
        print(f"  WER: {metrics['wer']:.4f}")
        print(f"  CER: {metrics['cer']:.4f}")
        print("  BLEU-1/2/3/4: " + " / ".join(f"{metrics[f'bleu{n}']:.2f}" for n in range(1, 5)))
        print(f"  ROUGE-L: {metrics['rouge_l']:.2f}")
        print(f"  chrF: {metrics['chrf']:.2f}")
        print(f"  Exact match: {metrics['exact_match_accuracy']:.4f}")
        print(f"  Token accuracy: {metrics['token_accuracy']:.4f}")
    else:
        print_metrics({'greedy': greedy_metrics, 'beam + LM' if lm else 'beam': metrics})
    print(f"  Empty predictions: {empty} ({100 * empty / max(1, len(predictions)):.1f}%)")
    print(f"  BLEU signature: {metrics['bleu_signature']}")
    if empty == len(predictions):
        print("  Every prediction is empty, which is what CTC collapsing to all-blank"
              " looks like. WER near 1.0 here means the model is not emitting tokens.")

    # The same predictions against other clips' sentences. A model that uses
    # the video scores far worse here; one that ignores it scores about the same.
    shuffled_control = None
    if len(targets) > 1:
        shuffled = shuffled_references(targets)
        decoded = {'greedy': greedy}
        if decoder == 'beam':
            decoded['beam + LM' if lm else 'beam'] = predictions
        shuffled_control = {name: {'wer': compute_wer(shuffled, texts),
                                   'cer': compute_cer(shuffled, texts)}
                            for name, texts in decoded.items()}
        print("\nThe same predictions scored against other clips' sentences:")
        for name, scores in shuffled_control.items():
            print(f"  {name:<10} WER {scores['wer']:.4f}  CER {scores['cer']:.4f}")
        print("  Scores close to the real ones mean the output barely depends on the video.")

    # Which tokens the output can detect, whatever the decoder makes of it
    names = output_units(config)
    presence = collected['presence']
    detection = token_detection(reference_ids(trainer, targets),
                                lambda token: presence[:, token].tolist(),
                                min_examples=DETECTION_MIN_CLIPS)
    if detection is None:
        print(f"\nToken detection: no token is in at least {DETECTION_MIN_CLIPS} clips and "
              f"missing from {DETECTION_MIN_CLIPS}, so there is nothing to score.")
    else:
        detection['per_token'] = sorted(
            ({'id': token, 'unit': names[token], **scores}
             for token, scores in detection['per_token'].items()),
            key=lambda entry: entry['auc'], reverse=True)
        print(f"\nToken detection, over the {detection['tokens_scored']} tokens in at least "
              f"{DETECTION_MIN_CLIPS} clips and missing from {DETECTION_MIN_CLIPS}:")
        print(f"  mean AUC {detection['mean_auc']:.3f} (chance 0.5; sentence length "
              f"alone gives {detection['mean_length_auc']:.3f})")
        print(f"  {detection['tokens_beating_length']} of {detection['tokens_scored']} tokens "
              f"are detected better than sentence length predicts them")
        print("  best detected: " + ", ".join(
            f"{entry['unit']} {entry['auc']:.2f}" for entry in detection['per_token'][:8]))

    print("\nExamples:")
    for target, prediction in list(zip(targets, predictions))[:5]:
        print(f"  target: {target[:90]}")
        print(f"  pred  : {prediction[:90]}")

    suffix = '' if decoder == 'greedy' else ('_beam_lm' if lm else '_beam')
    results_path = args.save_predictions or os.path.join(
        trainer.evaluation_dir,
        f"{args.split}_{os.path.splitext(os.path.basename(checkpoint_path))[0]}{suffix}.json")
    os.makedirs(os.path.dirname(os.path.abspath(results_path)), exist_ok=True)
    results = {
        'model': args.model,
        'config': args.config,
        'split': args.split,
        'checkpoint': checkpoint_path,
        'epoch': checkpoint.get('epoch'),
        'decoder': settings,
        'metrics': metrics,
        'token_detection': detection,
        'shuffled_control': shuffled_control,
    }
    if decoder == 'beam':
        results['greedy_metrics'] = greedy_metrics
    if lm_record is not None:
        results['lm_record'] = lm_record
    if tuning is not None:
        results['tuning'] = tuning
    results['pairs'] = [{'target': t, 'prediction': p, **({'greedy': g} if decoder == 'beam' else {})}
                        for t, p, g in zip(targets, predictions, greedy)]
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nWrote metrics and predictions to {results_path}")


if __name__ == "__main__":
    main()
