"""
Prepare scientific papers dataset for fine-tuning.

Downloads REAL arXiv paper abstracts (scientific domain) to create a dataset
for fine-tuning the TinyStories model. This creates a strong domain shift
from children's stories to academic/scientific writing.

Dataset: arXiv papers from HuggingFace 'scientific_papers' dataset
Size: We'll use ~20k abstracts for fine-tuning (enough for domain adaptation)
"""

import os
import pickle
import numpy as np
import tiktoken
from tqdm import tqdm


def download_arxiv_abstracts(output_file: str = 'arxiv_abstracts.txt', num_samples: int = 5000):
    """
    Download REAL arXiv abstracts using arXiv API.

    Queries the arXiv API directly to download real scientific paper abstracts
    from various categories (cs, physics, math, etc.) - perfect contrast with TinyStories.

    Args:
        output_file: Where to save the downloaded text
        num_samples: Number of abstracts to download (default: 5k, API limit friendly)
    """
    print(f"Downloading {num_samples:,} REAL arXiv abstracts from arXiv API...")
    print("This will query the arXiv API directly...")

    try:
        import urllib.request
        import urllib.parse
        import time
        import xml.etree.ElementTree as ET
        import ssl
    except ImportError as e:
        print(f"\nERROR: Missing library: {e}")
        return False

    # Create SSL context to handle certificate issues on Windows
    ssl_context = ssl._create_unverified_context()

    abstracts = []
    batch_size = 100  # arXiv API allows max ~1000 per request, we use 100 for safety
    max_retries = 3

    # arXiv categories to sample from (diverse scientific domains)
    categories = [
        'cs.AI',  # Artificial Intelligence
        'cs.LG',  # Machine Learning
        'cs.CL',  # Computation and Language
        'physics.comp-ph',  # Computational Physics
        'math.CO',  # Combinatorics
        'q-bio.NC',  # Neurons and Cognition
        'stat.ML',  # Machine Learning (Stats)
    ]

    print(f"\nQuerying arXiv API for papers from categories:")
    for cat in categories:
        print(f"  - {cat}")
    print()

    # Query arXiv API
    # API documentation: https://info.arxiv.org/help/api/index.html
    base_url = 'http://export.arxiv.org/api/query?'

    papers_per_category = num_samples // len(categories)

    for category in categories:
        print(f"Downloading {papers_per_category} abstracts from {category}...")

        for start in range(0, papers_per_category, batch_size):
            current_batch = min(batch_size, papers_per_category - start)

            # Build query URL
            query = f'search_query=cat:{category}'
            params = {
                'start': start,
                'max_results': current_batch,
                'sortBy': 'submittedDate',
                'sortOrder': 'descending'
            }

            url = base_url + query + '&' + urllib.parse.urlencode(params)

            # Make request with retry logic
            for attempt in range(max_retries):
                try:
                    with urllib.request.urlopen(url, context=ssl_context) as response:
                        xml_data = response.read().decode('utf-8')

                    # Parse XML
                    root = ET.fromstring(xml_data)
                    ns = {'atom': 'http://www.w3.org/2005/Atom'}

                    # Extract abstracts
                    entries = root.findall('atom:entry', ns)

                    if not entries:
                        print(f"  Warning: No papers found for {category} at offset {start}")
                        break

                    for entry in entries:
                        abstract = entry.find('atom:summary', ns)
                        if abstract is not None and abstract.text:
                            # Clean up abstract (remove extra whitespace, newlines)
                            clean_abstract = ' '.join(abstract.text.split())
                            if len(clean_abstract) > 100:
                                abstracts.append(clean_abstract)

                    print(f"  Downloaded {start + len(entries)}/{papers_per_category} from {category}")

                    # Be nice to the API - rate limit
                    time.sleep(1)
                    break

                except Exception as e:
                    print(f"  Attempt {attempt + 1}/{max_retries} failed: {e}")
                    if attempt < max_retries - 1:
                        time.sleep(2)
                    else:
                        print(f"  Failed to download batch after {max_retries} attempts")

    if not abstracts:
        print("\nERROR: No abstracts downloaded!")
        return False

    print(f"\nTotal abstracts downloaded: {len(abstracts):,}")

    # Write to file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n\n'.join(abstracts))

    print(f"Saved to {output_file}")

    # Print sample
    print("\n" + "="*70)
    print("Sample abstract (first one):")
    print("="*70)
    print(abstracts[0][:500])
    if len(abstracts[0]) > 500:
        print("...")
    print("="*70)

    return True


def prepare_dataset():
    """
    Prepare scientific papers dataset:
    1. Download REAL arXiv abstracts from HuggingFace
    2. Tokenize with GPT-2 BPE (same as TinyStories training)
    3. Create train/val split
    4. Save as binary files
    """

    # Download data
    input_file = os.path.join(os.path.dirname(__file__), 'arxiv_abstracts.txt')

    if not os.path.exists(input_file):
        # Download ~5000 abstracts (API-friendly, enough for fine-tuning)
        success = download_arxiv_abstracts(input_file, num_samples=5000)
        if not success:
            print("\nFailed to download dataset. Exiting.")
            return

    # Read the data
    with open(input_file, 'r', encoding='utf-8') as f:
        data = f.read()

    print(f"Loaded {len(data):,} characters of scientific text")
    print(f"First 500 chars:\n{data[:500]}\n")

    # Tokenize with GPT-2 BPE (same tokenizer as TinyStories training)
    print("Tokenizing with GPT-2 BPE...")
    enc = tiktoken.get_encoding("gpt2")
    tokens = enc.encode_ordinary(data)
    print(f"Total tokens: {len(tokens):,}")

    # Vocab size
    vocab_size = enc.n_vocab
    print(f"Vocabulary size: {vocab_size:,}")

    # Train/val split (90/10)
    n = len(tokens)
    split_idx = int(0.9 * n)
    train_tokens = tokens[:split_idx]
    val_tokens = tokens[split_idx:]

    print(f"Train tokens: {len(train_tokens):,}")
    print(f"Val tokens: {len(val_tokens):,}")

    # Save as binary files (uint16 since vocab_size < 65536)
    train_ids = np.array(train_tokens, dtype=np.uint16)
    val_ids = np.array(val_tokens, dtype=np.uint16)

    output_dir = os.path.dirname(__file__)
    train_file = os.path.join(output_dir, 'train.bin')
    val_file = os.path.join(output_dir, 'val.bin')

    train_ids.tofile(train_file)
    val_ids.tofile(val_file)

    print(f"\nSaved train.bin: {len(train_ids):,} tokens")
    print(f"Saved val.bin: {len(val_ids):,} tokens")

    # Save metadata (for compatibility with training script)
    meta = {
        'vocab_size': vocab_size,
        'tokenizer': 'gpt2',
    }

    meta_file = os.path.join(output_dir, 'meta.pkl')
    with open(meta_file, 'wb') as f:
        pickle.dump(meta, f)

    print(f"Saved meta.pkl")
    print("\nDataset preparation complete!")
    print(f"\nTo fine-tune, use: dataset='scientific_papers' in your config")

    # Print sample tokens for verification
    print("\n" + "="*70)
    print("Sample decoded text from train set (first 200 tokens):")
    print("="*70)
    sample_text = enc.decode(train_tokens[:200])
    print(sample_text)
    print("="*70)


if __name__ == '__main__':
    prepare_dataset()
