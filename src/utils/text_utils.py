import json
import re
import pandas as pd
from collections import Counter

def normalize_text(text: str) -> str:
    """
    Apply the same normalization used during vocabulary building
    - Lowercase (for Latin script)
    - Remove control characters and invisible formatting
    - Map special Unicode characters to standard forms
    - Collapse multiple spaces
    """

    # Step 1: Convert text to lowercase first (for Latin alphabets)
    text = text.lower()
    
    # STEP 2: Remove control characters and invisible formatting marks
    unwanted_pattern = re.compile(r'[\u0000-\u001F\u007F-\u009F\u200B-\u200F\u202A-\u202E\uFEFF]')
    text = unwanted_pattern.sub('', text)
    
    # STEP 3: Normalize special characters
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

        '\u09F7': '\u0964',  # Normalize Bengali currency numberator one to standard danda
    }

    for old_char, new_char in normalization_map.items():
        text = text.replace(old_char, new_char)

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
    
    # text = ''.join(allowed_pattern.findall(text))
    
    # Final cleanup: Collapse multiple spaces/tabs/newlines into a single space
    text = re.sub(r'\s+', ' ', text)
    
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
