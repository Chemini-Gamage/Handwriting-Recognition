import json

def parse_words_file(filepath):
    words = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            parts = line.split()
            if len(parts) < 9:
                continue
            
            try:
                word_data = {
                    'word_id': parts[0],
                    'segmentation': parts[1],
                    'graylevel': int(parts[2]),
                    'bbox': {
                        'x': int(parts[3]),
                        'y': int(parts[4]),
                        'w': int(parts[5]),
                        'h': int(parts[6])
                    },
                    'pos_tag': parts[7],
                    'transcription': " ".join(parts[8:])
                }
                words.append(word_data)
            except ValueError:
                continue
    return words

    words = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            # Skip comments and empty lines
            if not line or line.startswith('#'):
                continue
            
            parts = line.split()
            if len(parts) < 10:
                # Skip lines that don't have enough fields
                continue
            
            try:
                word_data = {
                    'word_id': parts[0],
                    'segmentation': parts[1],
                    'graylevel': int(parts[2]),
                    'num_components': int(parts[3]),
                    'bbox': {
                        'x': int(parts[4]),
                        'y': int(parts[5]),
                        'w': int(parts[6]),
                        'h': int(parts[7])
                    },
                    'pos_tag': parts[8],
                    'transcription': " ".join(parts[9:])
                }
                words.append(word_data)
            except ValueError:
                # Skip lines with invalid integers or formatting
                continue
    return words

def save_as_json(data, json_filepath):
    with open(json_filepath, 'w') as f:
        json.dump(data, f, indent=4)

if __name__ == '__main__':
    words_file = r'D:\bluechip\Handwriting-Recognition\data\iam_words\words.txt'
         # Input file path
    output_json = 'words_data.json'   # Output JSON file path
    
    print(f"Parsing file: {words_file}")
    words_data = parse_words_file(words_file)
    print(f"Parsed {len(words_data)} words.")
    
    print(f"Saving to JSON file: {output_json}")
    save_as_json(words_data, output_json)
    print("Done.")
