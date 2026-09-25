import json, re, os, hashlib, unicodedata
import pandas as pd
from collections import Counter
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, normalizers

def normalize_text(text: str) -> str:
    """
    Apply the same normalization used during vocabulary building
    - Lowercase (for Latin script)
    - Remove control characters and invisible formatting
    - Map special Unicode characters to standard forms
    - Collapse multiple spaces
    """

    # Unicode normalization form KC
    text = unicodedata.normalize('NFKC', text)

    # Lowercase, which only affects Latin script
    text = text.lower()

    # Remove control characters and invisible formatting marks
    unwanted_pattern = re.compile(r'[\u0000-\u001F\u007F-\u009F\u200B-\u200F\u202A-\u202E\uFEFF]')
    text = unwanted_pattern.sub('', text)

    # Map special characters to standard forms
    normalization_map = {
        # Spaces
        '\u00A0': ' ',   # Non-breaking space -> regular space

        # Quotes and dashes (Latin script)
        '\u2018': "'",      # Left single quote
        '\u2019': "'",      # Right single quote
        '\u201C': '"',      # Left double quote
        '\u201D': '"',      # Right double quote
        '\u2013': '-',      # En dash
        '\u2014': '-',      # Em dash
        # '\u2026': '...',    # Horizontal ellipsis

        '\u09F7': '\u0964',  # Normalize Bengali currency numberator one to standard danda
    }

    for old_char, new_char in normalization_map.items():
        text = text.replace(old_char, new_char)

    # Optional whitelist: Bangla, Latin letters, digits, basic punctuation, whitespace
    # allowed_pattern = re.compile(
    #     r'[\u0980-\u09FF]|'  # Bangla characters
    #     r'[a-z]|'            # English letters
    #     r'[0-9]|'            # Numbers
    #     r'[\.\,\?\!\"\'\-\:\;\(\)\[\]]|'  # Basic punctuation
    #     r'[\s]'              # Whitespace (space, tab, newline)
    # )

    # text = ''.join(allowed_pattern.findall(text))

    # Collapse runs of whitespace into a single space
    text = re.sub(r'\s+', ' ', text).strip()

    return text

def build_vocab_from_csv(csv_path, output_path):
    """Build character vocabulary from normalized text in CSV"""
    df = pd.read_csv(csv_path)

    # Normalize all texts
    all_text = ' '.join(df['text'].astype(str).apply(normalize_text).tolist())

    # Count characters and create vocabulary
    counter = Counter(all_text)
    vocab = sorted(counter.keys())

    # Diagnostic check for problematic characters
    problematic = [c for c in vocab if ord(c) < 32 and c not in ['\n', '\t', ' ']]
    if problematic:
        print(f"Warning: Found {len(problematic)} potentially problematic characters")
        for char in problematic:
            print(f"  U+{ord(char):04X}: {repr(char)}")

    # Create mapping with blank token at index 0 for CTC
    special_tokens = ['<blank>']
    char_to_id = {tok: idx for idx, tok in enumerate(special_tokens)}
    id_to_char = {idx: tok for tok, idx in char_to_id.items()}

    for idx, char in enumerate(vocab, start=len(special_tokens)):
        char_to_id[char] = idx
        id_to_char[idx] = char

    # Save vocabulary
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({'char_to_id': char_to_id, 'id_to_char': id_to_char},
                  f, ensure_ascii=False, indent=2)

    print(f"Vocabulary created with {len(char_to_id)} tokens")
    print("Characters:", repr(''.join(vocab)))

    return char_to_id, id_to_char

def text_to_int(sentence: str, char_to_id: dict) -> list:
    """Convert normalized text to integer sequence"""

    # Normalize the input sentence first
    normalized = normalize_text(sentence)

    # Map each character; skip any that are still not in vocab (shouldn't happen)
    return [char_to_id[char] for char in normalized if char in char_to_id]

def int_to_text(int_sequence, id_to_char):
    """Convert integer sequence back to text"""
    return ''.join([id_to_char[idx] for idx in int_sequence if idx in id_to_char])

def train_bpe_tokenizer(csv_path, output_path, vocab_size=1000, min_frequency=2):
    """
    Train a BPE tokenizer on the text column of the CSV
    """
    df = pd.read_csv(csv_path)

    # Normalize all texts using the normalize_text function
    texts = df['text'].astype(str).apply(normalize_text).tolist()

    # Located next to output_path rather than the CWD, which on Kaggle is not
    # the repo root
    output_dir = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(output_dir, exist_ok=True)
    temp_file = os.path.join(output_dir, "temp_texts_for_bpe.txt")
    with open(temp_file, "w", encoding="utf-8") as f:
        for t in texts:
            f.write(t + "\n")

    # Initialize BPE tokenizer
    tokenizer = Tokenizer(models.BPE())

    # Metaspace encodes each space as a marker on the following token, so the
    # token sequence records where words begin. A whitespace pre-tokenizer
    # discards that, and word boundaries are then unrecoverable from a decoded
    # CTC output: a memorised sentence still comes back split into sub-words.
    # The matching decoder turns the markers back into spaces.
    tokenizer.pre_tokenizer = pre_tokenizers.Metaspace()
    tokenizer.decoder = decoders.Metaspace()

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=[] # No special tokens, then after encoding, add 1 to shift index to 1..N, leaving 0 for blank
    )

    tokenizer.train([temp_file], trainer)

    # Clean up temp file
    os.remove(temp_file)

    # Decoding is what WER and CER are computed on, so a tokenizer that cannot
    # round-trip its own training text would make every reported score wrong
    failures = [t for t in texts[:2000] if tokenizer.decode(tokenizer.encode(t).ids) != t]
    if failures:
        raise RuntimeError(
            f"BPE tokenizer does not round-trip {len(failures)} of 2000 sentences; "
            f"first mismatch: {failures[0]!r} -> "
            f"{tokenizer.decode(tokenizer.encode(failures[0]).ids)!r}")

    # Save tokenizer
    tokenizer.save(output_path)

    print(f"BPE tokenizer saved to {output_path} with vocab size {tokenizer.get_vocab_size()}")
    return tokenizer

def load_bpe_tokenizer(tokenizer_path):
    """Load saved BPE tokenizer"""
    tokenizer = Tokenizer.from_file(tokenizer_path)
    if tokenizer.decoder is None:
        raise RuntimeError(
            f"{tokenizer_path} has no decoder, so decoding joins sub-word tokens "
            f"with spaces and inflates WER. Retrain it with "
            f"scripts/train_bpe_tokenizer.py.")
    return tokenizer

def label_fingerprint(config):
    """Short hash of what each output id means under this config.

    Stored in checkpoints so they are never used with a tokenizer whose ids
    stand for different tokens. The class count cannot catch that on its own: a
    retrained tokenizer of the same size loads without error and decodes
    nonsense.
    """
    if getattr(config, 'tokenization_type', 'character') == 'bpe':
        with open(config.bpe_tokenizer_path, 'r', encoding='utf-8') as f:
            saved = json.load(f)
        # Vocabulary, merges and word splitting fix the ids; the decoder only
        # affects how they are joined back into text
        content = {key: saved.get(key) for key in ('model', 'pre_tokenizer', 'normalizer')}
    else:
        with open(config.vocab_path, 'r', encoding='utf-8') as f:
            content = json.load(f)['char_to_id']
    blob = json.dumps(content, sort_keys=True, ensure_ascii=False).encode('utf-8')
    return hashlib.sha256(blob).hexdigest()[:16]

def label_fingerprint_problem(checkpoint, expected, path):
    """Explain why a checkpoint does not match the current labels, or return None."""
    saved = checkpoint.get('label_fingerprint')
    if saved is None:
        return (f"{path} has no tokenizer fingerprint, so it cannot be confirmed to "
                f"use the current tokenizer. It predates the fingerprint and "
                f"probably the Metaspace tokenizer too.")
    if saved != expected:
        return (f"{path} was trained with a different tokenizer or vocabulary "
                f"(fingerprint {saved}, current {expected}), so its output ids stand "
                f"for different tokens.")
    return None

def text_to_bpe_ids(text, tokenizer, blank_id=0):
    """
    Convert normalized text to BPE token IDs, shifting by 1 to reserve 0 for blank
    """
    normalized = normalize_text(text)
    encoded = tokenizer.encode(normalized)
    # Shift IDs by +1 to reserve 0 for blank
    return [id + 1 for id in encoded.ids]

def bpe_ids_to_text(ids, tokenizer, blank_id=0):
    """
    Convert BPE token IDs (with blank=0) back to text
    """
    # Shift back
    original_ids = [id - 1 for id in ids if id != blank_id]
    return tokenizer.decode(original_ids)
