"""Unit n-gram language model for CTC beam search.

The estimate is interpolated modified Kneser-Ney (Chen and Goodman, 1998), the
default in KenLM and SRILM, saved as a standard ARPA file. It is trained on the
units a CTC model emits (BPE tokens or characters), and every unit in the
model's vocabulary gets a probability, so nothing the model can emit is ever
out of vocabulary.

Only the standard library is used.
"""
import math
import os
from collections import defaultdict

BOS = '<s>'
EOS = '</s>'

# ARPA files store base-10 logs; scoring uses natural logs, like the CTC output
_LN10 = math.log(10.0)
# How ARPA writes log10(0). Used for <s>, which is only ever a context.
_ARPA_ZERO = -99.0


class NgramLM:
    """Scores a unit given the units before it. All scores are natural logs."""

    def __init__(self, order, logprobs, backoffs):
        self.order = order
        self.logprobs = logprobs    # n-gram -> log p(its last unit | the rest)
        self.backoffs = backoffs    # context -> log backoff weight
        self.vocabulary = {gram[0] for gram in logprobs if len(gram) == 1}

    def initial_context(self):
        return (BOS,)

    def advance(self, context, unit):
        """The context after `unit` has been emitted in `context`."""
        keep = self.order - 1
        return (context + (unit,))[-keep:] if keep else ()

    def score(self, context, unit):
        """log p(unit | context), backing off to shorter contexts as ARPA defines."""
        keep = self.order - 1
        context = context[-keep:] if keep else ()
        backoff = 0.0
        while True:
            logprob = self.logprobs.get(context + (unit,))
            if logprob is not None:
                return backoff + logprob
            if not context:
                raise KeyError(f"{unit!r} is not in the language model's vocabulary")
            backoff += self.backoffs.get(context, 0.0)
            context = context[1:]


def _check_unit(unit):
    if not isinstance(unit, str) or not unit or unit in (BOS, EOS) or any(c.isspace() for c in unit):
        raise ValueError(f"{unit!r} cannot be a language-model unit: units must be "
                         f"non-empty, contain no whitespace and not be {BOS} or {EOS}")


# KenLM's --discount_fallback values, for orders whose counts of counts cannot
# give an estimate. That happens when every n-gram of an order occurs often,
# e.g. the unigrams of a character model, and there the discount barely matters.
FALLBACK_DISCOUNTS = (0.5, 1.0, 1.5)


def _discounts(counts_of_counts):
    """Modified Kneser-Ney discounts D1, D2, D3+ from how many n-grams occur 1-4 times.

    Returns None when the counts cannot give valid discounts.
    """
    t1, t2, t3, t4 = counts_of_counts
    if min(counts_of_counts) == 0:
        return None
    y = t1 / (t1 + 2 * t2)
    discounts = (1 - 2 * y * t2 / t1, 2 - 3 * y * t3 / t2, 3 - 4 * y * t4 / t3)
    if not all(0 < value <= k for k, value in enumerate(discounts, start=1)):
        return None
    return discounts


def train_kneser_ney(sentences, vocabulary, order):
    """Estimate an interpolated modified Kneser-Ney model.

    `sentences` are lists of units, and `vocabulary` is every unit the CTC model
    can emit; units that never occur in the text still get a share of the
    unigram mass. Returns the model and a summary of the estimate.
    """
    if order < 2:
        raise ValueError(f"order must be at least 2, got {order}")
    vocabulary = set(vocabulary)
    for unit in vocabulary:
        _check_unit(unit)

    # Raw counts of every n-gram up to `order` that predicts a unit or </s>.
    # <s> is never predicted, so no n-gram ends with it.
    raw = [None] + [defaultdict(int) for _ in range(order)]
    for number, sentence in enumerate(sentences):
        unknown = [unit for unit in sentence if unit not in vocabulary]
        if unknown:
            raise ValueError(f"Sentence {number} has units outside the vocabulary: {unknown[:5]}")
        padded = [BOS] + list(sentence) + [EOS]
        for end in range(1, len(padded)):
            for n in range(1, min(order, end + 1) + 1):
                raw[n][tuple(padded[end - n + 1:end + 1])] += 1

    # Kneser-Ney replaces a lower-order count with the number of different
    # units seen before that n-gram. N-grams that start at <s> cannot have one,
    # so they keep their real count.
    adjusted = [None] * (order + 1)
    adjusted[order] = dict(raw[order])
    for n in range(order - 1, 0, -1):
        left = defaultdict(int)
        for gram in raw[n + 1]:
            left[gram[1:]] += 1
        adjusted[n] = {gram: count if gram[0] == BOS else left[gram]
                       for gram, count in raw[n].items()}

    discounts = [None] * (order + 1)
    fallback = {}
    for n in range(1, order + 1):
        counts_of_counts = [0, 0, 0, 0]
        for count in adjusted[n].values():
            if count <= 4:
                counts_of_counts[count - 1] += 1
        discounts[n] = _discounts(counts_of_counts)
        if discounts[n] is None:
            discounts[n] = FALLBACK_DISCOUNTS
            fallback[n] = counts_of_counts

    # Lowest order first, because each order interpolates with the one below.
    # The unigrams interpolate with a uniform distribution over every unit
    # and </s>.
    predictable = vocabulary | {EOS}
    probs, backoffs = {}, {}
    for n in range(1, order + 1):
        d = discounts[n]
        totals = defaultdict(int)
        kinds = defaultdict(lambda: [0, 0, 0])
        for gram, count in adjusted[n].items():
            totals[gram[:-1]] += count
            kinds[gram[:-1]][min(count, 3) - 1] += 1
        # The probability mass the discounts free up in each context, which
        # goes to the lower order
        weights = {context: sum(di * ki for di, ki in zip(d, kinds[context])) / total
                   for context, total in totals.items()}

        for gram, count in adjusted[n].items():
            context = gram[:-1]
            lower = 1 / len(predictable) if n == 1 else probs[gram[1:]]
            probs[gram] = (count - d[min(count, 3) - 1]) / totals[context] + weights[context] * lower
        if n == 1:
            for unit in predictable:
                probs.setdefault((unit,), weights[()] / len(predictable))
        else:
            backoffs.update(weights)

    logprobs = {gram: math.log(p) for gram, p in probs.items()}
    log_backoffs = {context: math.log(w) for context, w in backoffs.items()}
    summary = {
        'order': order,
        'ngrams': {n: len(adjusted[n]) for n in range(1, order + 1)},
        'discounts': {n: [round(x, 4) for x in discounts[n]] for n in range(1, order + 1)},
        # Orders that used FALLBACK_DISCOUNTS, with the counts of n-grams seen
        # 1-4 times that could not give an estimate
        'discount_fallback': fallback,
    }
    return NgramLM(order, logprobs, log_backoffs), summary


def perplexity(lm, sentences):
    """Per-unit perplexity of `sentences`, counting the end of each sentence."""
    total, count = 0.0, 0
    for sentence in sentences:
        context = lm.initial_context()
        for unit in list(sentence) + [EOS]:
            total += lm.score(context, unit)
            count += 1
            context = lm.advance(context, unit)
    if count == 0:
        raise ValueError("No sentences to measure perplexity on")
    return math.exp(-total / count)


def metadata_path(arpa_path):
    """Where the record of what an ARPA file was trained on is kept.

    ARPA readers such as KenLM require the file to start with \\data\\, so the
    record is a JSON file beside it rather than a header.
    """
    return os.path.splitext(arpa_path)[0] + '.json'


def write_arpa(lm, path):
    by_order = defaultdict(list)
    for gram in lm.logprobs:
        by_order[len(gram)].append(gram)
    # <s> has a backoff weight but, never being predicted, no probability
    only_context = [context for context in lm.backoffs if context not in lm.logprobs]
    if any(context != (BOS,) for context in only_context):
        raise ValueError(f"Contexts without a probability other than {BOS}: {only_context[:5]}")
    start = (BOS,) in lm.backoffs

    with open(path, 'w', encoding='utf-8') as f:
        f.write('\\data\\\n')
        for n in range(1, lm.order + 1):
            f.write(f"ngram {n}={len(by_order[n]) + (1 if n == 1 and start else 0)}\n")
        for n in range(1, lm.order + 1):
            f.write(f"\n\\{n}-grams:\n")
            if n == 1 and start:
                f.write(f"{_ARPA_ZERO:.6f}\t{BOS}\t{lm.backoffs[(BOS,)] / _LN10:.6f}\n")
            for gram in sorted(by_order[n]):
                line = f"{lm.logprobs[gram] / _LN10:.6f}\t{' '.join(gram)}"
                if gram in lm.backoffs:
                    line += f"\t{lm.backoffs[gram] / _LN10:.6f}"
                f.write(line + '\n')
        f.write('\n\\end\\\n')


def read_arpa(path):
    logprobs, backoffs = {}, {}
    declared, seen = {}, defaultdict(int)
    section = None
    with open(path, 'r', encoding='utf-8') as f:
        for number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            if line == '\\data\\':
                section = 'data'
            elif line == '\\end\\':
                break
            elif line.startswith('\\') and line.endswith('-grams:'):
                section = int(line[1:-len('-grams:')])
            elif section == 'data' and line.startswith('ngram '):
                n, count = line[len('ngram '):].split('=')
                declared[int(n)] = int(count)
            elif isinstance(section, int):
                fields = line.split()
                if len(fields) not in (section + 1, section + 2):
                    raise ValueError(f"{path}:{number}: expected a {section}-gram line, got {line!r}")
                gram = tuple(fields[1:section + 1])
                seen[section] += 1
                logprob = float(fields[0])
                if logprob > _ARPA_ZERO:
                    logprobs[gram] = logprob * _LN10
                if len(fields) == section + 2:
                    backoffs[gram] = float(fields[-1]) * _LN10
            else:
                raise ValueError(f"{path}:{number}: unexpected line {line!r}")
        else:
            raise ValueError(f"{path} ends without \\end\\, so it is incomplete")

    if not declared or dict(seen) != declared:
        raise ValueError(f"{path} declares n-gram counts {declared} but contains {dict(seen)}")
    return NgramLM(max(declared), logprobs, backoffs)
