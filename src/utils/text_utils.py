import json
import pandas as pd
from collections import Counter

def build_vocab_from_csv(csv_path, output_path):
    """Builds character vocabulary from text in CSV and saves it."""
    df = pd.read_csv(csv_path)
    all_text = ' '.join(df['text'].astype(str).tolist())
    
    # Count characters and create vocabulary
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
    return char_to_id, id_to_char

def text_to_int(sentence, char_to_id):
    """Convert text to integer sequence."""
    return [char_to_id[char] for char in sentence if char in char_to_id]

def int_to_text(int_sequence, id_to_char):
    """Convert integer sequence back to text."""
    return ''.join([id_to_char[idx] for idx in int_sequence if idx in id_to_char])
