"""Train the n-gram language model that CTC beam search uses.

It is trained on the units a CTC model emits (BPE tokens or characters), from
the train split's sentences. Any sentence that also appears in val or test is
left out, so neither tuning on val nor scoring on test benefits from it.

    python scripts/train_ngram_lm.py --config subset
    python scripts/train_ngram_lm.py --model cnn_bilstm --order 6
    python scripts/train_ngram_lm.py --config subset --extra-text bangla_corpus.txt

The subset config's csv_path is the full dataset, so the subset model's LM
learns from every train sentence. That is text alone, with no val or test
sentence in it, and it is what the full model's LM will use too.

Writes <output_dir>/lm/<units>_<order>gram.arpa and a .json beside it recording
what it was trained on. Pass the .arpa to scripts/evaluate.py with --lm.
"""
import os
import sys
import json
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from src.utils.environment import check_environment
from src.utils.text_utils import (normalize_text, load_bpe_tokenizer, text_to_bpe_ids,
                                  text_to_int, output_units, label_fingerprint)
from src.utils.ngram_lm import train_kneser_ney, perplexity, write_arpa, metadata_path

# A BPE token is about three characters, so four of them span a word or two
DEFAULT_ORDER = {'bpe': 4, 'character': 6}

# Orders whose val perplexity is reported, to help choose --order
COMPARED_ORDERS = (2, 3, 4, 5, 6)


def main():
    parser = argparse.ArgumentParser(description="Train the n-gram LM for CTC beam search")
    parser.add_argument('--model', choices=['vivit', 'cnn_bilstm'], default='vivit')
    parser.add_argument('--config', choices=['prod', 'test', 'overfit', 'subset'], default='prod',
                        help="Config whose tokenizer, split and paths are used")
    parser.add_argument('--csv', default=None,
                        help="Dataset CSV (defaults to the config's csv_path)")
    parser.add_argument('--order', type=int, default=None,
                        help="n-gram order (default 4 for BPE, 6 for characters)")
    parser.add_argument('--extra-text', default=None,
                        help="More training text, one sentence per line, e.g. a Bangla corpus")
    parser.add_argument('--output', default=None,
                        help="ARPA file to write (defaults to <output_dir>/lm/<units>_<order>gram.arpa)")
    args = parser.parse_args()
    check_environment()

    if args.model == 'vivit':
        if args.config == 'test':
            from configs.test_vivit_ctc_config import config
        elif args.config == 'overfit':
            from configs.overfit_vivit_ctc_config import config
        elif args.config == 'subset':
            from configs.subset_vivit_ctc_config import config
        else:
            from configs.vivit_ctc_config import config
    else:
        if args.config in ('overfit', 'subset'):
            parser.error("CNN-BiLSTM has only the prod and test configs")
        if args.config == 'test':
            from configs.test_cnn_bilstm_ctc_config import config
        else:
            from configs.cnn_bilstm_ctc_config import config

    tokenization = getattr(config, 'tokenization_type', 'character')
    order = args.order or DEFAULT_ORDER[tokenization]
    csv_path = args.csv or config.csv_path

    df = pd.read_csv(csv_path)
    with open(config.train_val_test_split_path, 'r', encoding='utf-8') as f:
        split = json.load(f)
    df['split'] = df['recording'].map(split)
    unassigned = int(df['split'].isna().sum())
    if unassigned:
        raise RuntimeError(f"{unassigned} of {len(df)} recordings in {csv_path} are not in "
                           f"{config.train_val_test_split_path}. Regenerate the split or fix the CSV.")
    df['norm'] = df['text'].astype(str).map(normalize_text)

    heldout = set(df.loc[df['split'] != 'train', 'norm'])
    train_texts = df.loc[df['split'] == 'train', 'norm'].tolist()
    texts = [text for text in train_texts if text not in heldout]
    excluded = len(train_texts) - len(texts)

    extra_count = extra_excluded = 0
    if args.extra_text:
        with open(args.extra_text, 'r', encoding='utf-8') as f:
            extra = [normalize_text(line) for line in f]
        extra = [text for text in extra if text]
        extra_count = sum(1 for text in extra if text not in heldout)
        extra_excluded = len(extra) - extra_count
        texts += [text for text in extra if text not in heldout]

    if tokenization == 'bpe':
        tokenizer = load_bpe_tokenizer(config.bpe_tokenizer_path)
        encode = lambda text: text_to_bpe_ids(text, tokenizer)
    else:
        with open(config.vocab_path, 'r', encoding='utf-8') as f:
            char_to_id = json.load(f)['char_to_id']
        encode = lambda text: text_to_int(text, char_to_id)
    units = output_units(config)
    sentences = [[units[i] for i in encode(text)] for text in texts]
    empty = sum(1 for sentence in sentences if not sentence)
    if empty:
        raise RuntimeError(f"{empty} training sentences encode to no units at all, so the "
                           f"text and the {tokenization} vocabulary do not match.")
    val_sentences = [[units[i] for i in encode(text)]
                     for text in df.loc[df['split'] == 'val', 'norm']]

    print(f"=== {tokenization.upper()} n-gram language model ===")
    print(f"Train sentences: {len(train_texts)}, of which {excluded} also appear in val or "
          f"test and are left out")
    if args.extra_text:
        print(f"Extra text from {args.extra_text}: {extra_count} sentences "
              f"({extra_excluded} left out for appearing in val or test)")
    print(f"Training on {len(sentences)} sentences, {sum(map(len, sentences))} units, "
          f"vocabulary {len(units) - 1}")

    print("\nVal perplexity by order (lower is better):")
    for n in sorted(set(COMPARED_ORDERS) | {order}):
        candidate, candidate_summary = train_kneser_ney(sentences, units[1:], n)
        candidate_perplexity = perplexity(candidate, val_sentences)
        print(f"  {n}-gram: {candidate_perplexity:.2f}" +
              ("   <- writing this one" if n == order else ""))
        if n == order:
            lm, summary, val_perplexity = candidate, candidate_summary, candidate_perplexity
    print(f"\n{order}-gram counts: {summary['ngrams']}")
    print(f"Discounts (D1, D2, D3+): {summary['discounts']}")
    if summary['discount_fallback']:
        print(f"Orders {sorted(summary['discount_fallback'])} could not estimate discounts from "
              f"their counts of counts and use the fixed fallback (0.5, 1.0, 1.5)")

    output = args.output or os.path.join(
        config.output_dir, 'lm',
        f"{tokenization}_{order}gram{'_extra' if args.extra_text else ''}.arpa")
    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    write_arpa(lm, output)
    record = {
        'units': tokenization,
        'label_fingerprint': label_fingerprint(config),
        'order': order,
        'csv': csv_path,
        'split_file': config.train_val_test_split_path,
        'sentences': len(sentences),
        'train_sentences': len(texts) - extra_count,
        'excluded_heldout': excluded,
        'extra_text': args.extra_text,
        'extra_sentences': extra_count,
        'extra_excluded_heldout': extra_excluded,
        'val_perplexity': val_perplexity,
        'ngrams': summary['ngrams'],
        'discounts': summary['discounts'],
        'discount_fallback': summary['discount_fallback'],
    }
    with open(metadata_path(output), 'w', encoding='utf-8') as f:
        json.dump(record, f, ensure_ascii=False, indent=2)
    print(f"\nWrote {output} ({os.path.getsize(output) / 1024 ** 2:.0f} MB) and "
          f"{metadata_path(output)}")


if __name__ == "__main__":
    main()
