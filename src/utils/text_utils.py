import json
import pandas as pd
from collections import Counter

def build_vocab_from_csv(csv_path, output_path):
    """Builds character vocabulary from text in CSV and saves it."""
    df = pd.read_csv(csv_path)
    
    # NORMALIZE: Convert all text to lowercase first (for Latin alphabets)
    all_text = ' '.join(df['text'].astype(str).str.lower().tolist())

    # NORMALIZE: Replace non-breaking spaces with regular spaces
    all_text = all_text.replace('\u00A0', ' ')  # Replace non-breaking space

    # COMPREHENSIVE NORMALIZATION: Replace various space-like characters with regular spaces
    space_normalizations = {
        '\u00A0': ' ',  # Non-breaking space
        '\u200B': ' ',  # Zero width space
        '\u200C': ' ',  # Zero width non-joiner
        '\u200D': ' ',  # Zero width joiner
        '\u200E': ' ',  # Left-to-right mark
        '\u200F': ' ',  # Right-to-left mark
        '\u202A': ' ',  # Left-to-right embedding
        '\u202C': ' ',  # Pop directional formatting
        '\uFEFF': ' ',  # Zero width no-break space (BOM)
    }

    for old_char, new_char in space_normalizations.items():
        all_text = all_text.replace(old_char, new_char)
    
    # Remove other unwanted invisible characters but keep meaningful ones
    # Keep only: Bangla chars, English letters, numbers, punctuation, and regular whitespace
    # allowed_pattern = re.compile(
    #     r'[\u0980-\u09FF]|'  # Bangla characters
    #     r'[a-z]|'            # English letters (now lowercase only)
    #     r'[0-9]|'            # Numbers
    #     r'[\.\,\?\!\"\'\-\:\;\(\)]|'  # Basic punctuation
    #     r'\s'                 # Whitespace
    # )
    
    # filtered_text = ''.join(allowed_pattern.findall(all_text))
    
    # Count characters and create vocabulary
    # counter = Counter(filtered_text)
    counter = Counter(all_text)
    vocab = sorted(counter.keys())
    
    # Create mapping (add BLANK token at index 0 for CTC)
    char_to_id = {'<blank>': 0}
    id_to_char = {0: '<blank>'}
    
    for idx, char in enumerate(vocab, start=1):
        char_to_id[char] = idx
        id_to_char[idx] = char
    
    # Save vocabulary
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({'char_to_id': char_to_id, 'id_to_char': id_to_char}, 
                 f, ensure_ascii=False, indent=2)
    
    print(f"Vocabulary created with {len(char_to_id)} tokens")
    print("Characters:", ''.join(vocab))
    return char_to_id, id_to_char

def text_to_int(sentence, char_to_id):
    """Convert text to integer sequence."""
    return [char_to_id[char] for char in sentence if char in char_to_id]

def int_to_text(int_sequence, id_to_char):
    """Convert integer sequence back to text."""
    return ''.join([id_to_char[idx] for idx in int_sequence if idx in id_to_char])
