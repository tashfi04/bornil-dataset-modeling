import editdistance
from typing import List, Tuple
import numpy as np
from sacrebleu.metrics import BLEU, CHRF
from sacrebleu.tokenizers.tokenizer_intl import TokenizerV14International

# The 'intl' tokenizer splits off Unicode punctuation, including the Bangla
# danda (।). The default '13a' tokenizer does not, so 'হবে।' and 'হবে' would
# count as different words. Report the signature so scores can be reproduced.
BLEU_TOKENIZER = 'intl'
_intl_tokenize = TokenizerV14International()

def compute_wer(references: List[str], hypotheses: List[str]) -> float:
    """
    Compute Word Error Rate (WER)

    Args:
        references: List of reference sentences
        hypotheses: List of predicted sentences

    Returns:
        Word Error Rate
    """
    total_errors = 0
    total_words = 0

    for ref, hyp in zip(references, hypotheses):
        ref_words = ref.split()
        hyp_words = hyp.split()
        errors = editdistance.eval(ref_words, hyp_words)
        total_errors += errors
        total_words += len(ref_words)

    return total_errors / total_words if total_words > 0 else 0.0

def compute_cer(references: List[str], hypotheses: List[str]) -> float:
    """
    Compute Character Error Rate (CER)

    Args:
        references: List of reference sentences
        hypotheses: List of predicted sentences

    Returns:
        Character Error Rate
    """
    total_errors = 0
    total_chars = 0

    for ref, hyp in zip(references, hypotheses):
        errors = editdistance.eval(ref, hyp)
        total_errors += errors
        total_chars += len(ref)

    return total_errors / total_chars if total_chars > 0 else 0.0

def compute_accuracy(references: List[str], hypotheses: List[str]) -> Tuple[float, float]:
    """
    Compute exact match accuracy and token-level accuracy

    Args:
        references: List of reference sentences
        hypotheses: List of predicted sentences

    Returns:
        exact_match_accuracy, token_accuracy
    """
    exact_matches = 0
    total_tokens = 0
    correct_tokens = 0

    for ref, hyp in zip(references, hypotheses):
        # Exact match
        if ref == hyp:
            exact_matches += 1

        # Token-level accuracy (character level for text)
        ref_chars = list(ref)
        hyp_chars = list(hyp)

        # Compare up to min length
        min_len = min(len(ref_chars), len(hyp_chars))
        for i in range(min_len):
            if ref_chars[i] == hyp_chars[i]:
                correct_tokens += 1
        total_tokens += len(ref_chars)

    exact_match_acc = exact_matches / len(references) if references else 0.0
    token_acc = correct_tokens / total_tokens if total_tokens > 0 else 0.0

    return exact_match_acc, token_acc

def compute_bleu(references: List[str], hypotheses: List[str]) -> dict:
    """Corpus BLEU-1 to BLEU-4 on a 0-100 scale, plus the sacrebleu signature.

    BLEU-n uses n-grams up to order n with uniform weights, as reported in the
    sign language translation literature.
    """
    scores = {}
    for order in range(1, 5):
        bleu = BLEU(tokenize=BLEU_TOKENIZER, max_ngram_order=order)
        scores[f'bleu{order}'] = bleu.corpus_score(hypotheses, [references]).score
    scores['bleu_signature'] = str(bleu.get_signature())
    return scores

def compute_chrf(references: List[str], hypotheses: List[str]) -> float:
    """Corpus chrF on a 0-100 scale. Character n-grams suit Bangla's long words."""
    return CHRF().corpus_score(hypotheses, [references]).score

def _lcs_length(a: List[str], b: List[str]) -> int:
    previous = [0] * (len(b) + 1)
    for token_a in a:
        current = [0]
        for j, token_b in enumerate(b, 1):
            current.append(previous[j - 1] + 1 if token_a == token_b
                           else max(previous[j], current[j - 1]))
        previous = current
    return previous[-1]

def compute_rouge_l(references: List[str], hypotheses: List[str], beta: float = 1.2) -> float:
    """Sentence-averaged ROUGE-L F-score on a 0-100 scale.

    Follows the coco-caption formulation (beta = 1.2) used for sign language
    translation. Written here because the rouge-score package drops every
    non-Latin character during tokenization, which would erase Bangla text.
    """
    total = 0.0
    for ref, hyp in zip(references, hypotheses):
        ref_tokens = _intl_tokenize(ref).split()
        hyp_tokens = _intl_tokenize(hyp).split()
        lcs = _lcs_length(ref_tokens, hyp_tokens)
        if lcs == 0:
            continue
        precision = lcs / len(hyp_tokens)
        recall = lcs / len(ref_tokens)
        total += ((1 + beta ** 2) * precision * recall) / (recall + beta ** 2 * precision)
    return 100 * total / len(references) if references else 0.0

def calculate_all_metrics(references: List[str], hypotheses: List[str]) -> dict:
    """
    Calculate all evaluation metrics

    Returns:
        Dictionary with all metrics
    """
    wer = compute_wer(references, hypotheses)
    cer = compute_cer(references, hypotheses)
    exact_match_acc, token_acc = compute_accuracy(references, hypotheses)

    return {
        'wer': wer,
        'cer': cer,
        **compute_bleu(references, hypotheses),
        'rouge_l': compute_rouge_l(references, hypotheses),
        'chrf': compute_chrf(references, hypotheses),
        'exact_match_accuracy': exact_match_acc,
        'token_accuracy': token_acc,
        'num_samples': len(references)
    }
