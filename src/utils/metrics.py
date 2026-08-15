import editdistance
from typing import List, Tuple
import numpy as np

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
        'exact_match_accuracy': exact_match_acc,
        'token_accuracy': token_acc,
        'num_samples': len(references)
    }
