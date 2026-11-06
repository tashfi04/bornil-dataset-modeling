"""Whether a CTC model can tell which tokens a clip contains, regardless of decoding.

A model can carry real information about the signs in a clip and still decode
to almost nothing, because CTC only emits a token where it beats the blank.
This scores how likely each token is to be in a clip by its highest
probability anywhere in the clip, and measures for each token how well that
ranks the clips that contain it above those that do not (ROC AUC, where 0.5 is
chance). It is far more sensitive than WER when the model is weak.

A score that only tracks how long the sentence is would also beat chance,
since longer sentences contain more tokens, so the same AUC is reported for
the reference length itself as a baseline the model has to beat.

Only the standard library is used.
"""
from collections import Counter


def roc_auc(scores, labels):
    """Area under the ROC curve. Tied scores share their average rank."""
    positives = sum(1 for label in labels if label)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        raise ValueError("ROC AUC needs at least one positive and one negative example")

    order = sorted(range(len(scores)), key=lambda i: scores[i])
    rank_sum = 0.0
    start = 0
    while start < len(order):
        end = start
        while end + 1 < len(order) and scores[order[end + 1]] == scores[order[start]]:
            end += 1
        # Ranks start + 1 .. end + 1 are shared by the tied scores
        average_rank = (start + end) / 2 + 1
        rank_sum += average_rank * sum(1 for i in order[start:end + 1] if labels[i])
        start = end + 1
    return (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def token_detection(references, column, min_examples=10):
    """Per-token detection AUC over a set of clips.

    `references` holds each clip's reference ids, and `column(id)` returns
    that id's score for every clip, in the same order. Only tokens present in,
    and absent from, at least `min_examples` clips are scored, since an AUC
    from a handful of clips is noise.
    """
    clips = len(references)
    contained = [set(reference) for reference in references]
    present = Counter(token for tokens in contained for token in tokens)
    eligible = sorted(token for token, count in present.items()
                      if count >= min_examples and clips - count >= min_examples)
    if not eligible:
        return None

    lengths = [len(reference) for reference in references]
    per_token, model_aucs, length_aucs = {}, [], []
    for token in eligible:
        labels = [token in tokens for tokens in contained]
        model_auc = roc_auc(column(token), labels)
        length_auc = roc_auc(lengths, labels)
        per_token[token] = {'auc': model_auc, 'length_auc': length_auc,
                            'clips_with_token': present[token]}
        model_aucs.append(model_auc)
        length_aucs.append(length_auc)

    return {
        'tokens_scored': len(eligible),
        'mean_auc': sum(model_aucs) / len(model_aucs),
        'mean_length_auc': sum(length_aucs) / len(length_aucs),
        'tokens_beating_length': sum(1 for m, l in zip(model_aucs, length_aucs) if m > l),
        'min_examples': min_examples,
        'per_token': per_token,
    }
