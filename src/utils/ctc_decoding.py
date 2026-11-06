"""Turning CTC output into id sequences: greedy, or prefix beam search with an LM.

Only the standard library is used, so the search can be checked without torch.
"""
import heapq
import math
import multiprocessing

from src.utils.ngram_lm import EOS

NEG_INF = float('-inf')

# Set before the worker processes fork, so they inherit the language model
# instead of receiving a pickled copy with every clip
_search_settings = {}


def collapse(ids, blank=0):
    """CTC's collapse rule: merge repeated ids, then drop blanks."""
    decoded = []
    previous = None
    for idx in ids:
        if idx != blank and idx != previous:
            decoded.append(idx)
        previous = idx
    return decoded


def _log_add(a, b):
    if a == NEG_INF:
        return b
    if b == NEG_INF:
        return a
    if a < b:
        a, b = b, a
    return a + math.log1p(math.exp(b - a))


def prefix_beam_search(steps, beam_size, lm=None, units=None, lm_weight=0.0,
                       insertion_bonus=0.0, blank=0):
    """The likeliest id sequence under the CTC output and an optional LM.

    `steps` holds one entry per time step, (blank log-prob, [(id, log-prob),
    ...]), with the non-blank candidates already cut to the likeliest few. A
    prefix is ranked by log P_ctc + lm_weight * log P_lm + insertion_bonus *
    length, the usual shallow fusion; the bonus offsets the LM's preference
    for short outputs. `units` maps an output id to the unit the LM was
    trained on. End of sentence is scored once, after the last step.
    """
    if lm is not None and units is None:
        raise ValueError("Scoring with a language model needs `units`")

    # Each prefix keeps the probability of the paths ending in a blank and of
    # those ending in its last id, because only the latter merge with a repeat
    beams = {(): (0.0, NEG_INF)}
    lm_scores = {(): 0.0}
    contexts = {(): lm.initial_context() if lm is not None else None}

    def extend(prefix, token):
        new = prefix + (token,)
        if new not in lm_scores:
            if lm is None:
                lm_scores[new] = 0.0
            else:
                unit = units[token]
                lm_scores[new] = lm_scores[prefix] + lm.score(contexts[prefix], unit)
                contexts[new] = lm.advance(contexts[prefix], unit)
        return new

    def rank(prefix, probabilities):
        return (_log_add(*probabilities) + lm_weight * lm_scores[prefix]
                + insertion_bonus * len(prefix))

    for blank_logprob, candidates in steps:
        following = {}

        def add(prefix, ends_blank, ends_token):
            entry = following.get(prefix)
            if entry is None:
                following[prefix] = [ends_blank, ends_token]
            else:
                entry[0] = _log_add(entry[0], ends_blank)
                entry[1] = _log_add(entry[1], ends_token)

        for prefix, (ends_blank, ends_token) in beams.items():
            total = _log_add(ends_blank, ends_token)
            add(prefix, total + blank_logprob, NEG_INF)
            last = prefix[-1] if prefix else None
            for token, logprob in candidates:
                if token == blank:
                    raise ValueError("Candidates must not include the blank")
                if token == last:
                    # Without a blank in between, a repeat merges into the last id;
                    # after one, it is a new emission of the same id
                    add(prefix, NEG_INF, ends_token + logprob)
                    add(extend(prefix, token), NEG_INF, ends_blank + logprob)
                else:
                    add(extend(prefix, token), NEG_INF, total + logprob)

        kept = heapq.nlargest(beam_size, following.items(),
                              key=lambda item: rank(item[0], item[1]))
        beams = {prefix: tuple(probabilities) for prefix, probabilities in kept}

    def final(item):
        prefix, probabilities = item
        score = rank(prefix, probabilities)
        if lm is not None:
            score += lm_weight * lm.score(contexts[prefix], EOS)
        return score

    best, _ = max(beams.items(), key=final)
    return list(best)


def _search_one(steps):
    return prefix_beam_search(steps, **_search_settings)


def beam_search_all(clips, workers=1, **settings):
    """prefix_beam_search over many clips, split across `workers` processes.

    `clips` holds each clip's steps; `settings` are prefix_beam_search's other
    arguments. Results come back in the order of `clips`.
    """
    _search_settings.clear()
    _search_settings.update(settings)
    try:
        if workers <= 1 or len(clips) < 2 * workers:
            return [_search_one(steps) for steps in clips]
        with multiprocessing.get_context('fork').Pool(workers) as pool:
            return pool.map(_search_one, clips,
                            chunksize=max(1, len(clips) // (4 * workers)))
    finally:
        _search_settings.clear()
