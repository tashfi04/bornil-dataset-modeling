import json
import re
import pandas as pd
from collections import Counter

def build_vocab_from_csv(csv_path, output_path):
    """Builds character vocabulary from text in CSV and saves it."""
    df = pd.read_csv(csv_path)
    
    # NORMALIZE: Convert all text to lowercase first (for Latin alphabets)
    all_text = ' '.join(df['text'].astype(str).str.lower().tolist())

    # STEP 1: Remove control characters and invisible formatting marks
    unwanted_pattern = re.compile(r'[\u0000-\u001F\u007F-\u009F\u200B-\u200F\u202A-\u202E\uFEFF]')
    all_text = unwanted_pattern.sub('', all_text)

    # STEP 2: Normalize special characters
    # Convert to standard forms
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
    }

    for old_char, new_char in normalization_map.items():
        all_text = all_text.replace(old_char, new_char)

    # STEP 4: Filter to keep only meaningful characters (whitelist)
    # Remove other unwanted invisible characters but keep meaningful ones
    # Keep only: Bangla chars, English letters, numbers, punctuation, and regular whitespace
    # allowed_pattern = re.compile(
    #     r'[\u0980-\u09FF]|'  # Bangla characters
    #     r'[a-z]|'            # English letters
    #     r'[0-9]|'            # Numbers
    #     r'[\.\,\?\!\"\'\-\:\;\(\)\[\]]|'  # Basic punctuation
    #     r'[\s]'              # Whitespace (space, tab, newline)
    # )
    
    # filtered_text = ''.join(allowed_pattern.findall(all_text))

    # Final cleanup: collapse multiple spaces into single spaces
    all_text = re.sub(r'\s+', ' ', all_text)

    
    # Count characters and create vocabulary
    counter = Counter(all_text)
    vocab = sorted(counter.keys())

    
    # Diagnostic check for problematic characters
    problematic = [c for c in vocab if ord(c) < 32 and c not in ['\n', '\t', ' ']]
    if problematic:
        print(f"Warning: Found {len(problematic)} potentially problematic characters")
        for char in problematic:
            print(f"  U+{ord(char):04X}: {repr(char)}")


    
    # Create mapping (add BLANK token at index 0 for CTC)
    special_tokens = ['<blank>']  # extend if needed
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

def text_to_int(sentence, char_to_id):
    """Convert text to integer sequence."""
    return [char_to_id[char] for char in sentence if char in char_to_id]

def int_to_text(int_sequence, id_to_char):
    """Convert integer sequence back to text."""
    return ''.join([id_to_char[idx] for idx in int_sequence if idx in id_to_char])
