"""
Prepare scientific papers dataset using HuggingFace datasets (FAST!).

This version uses pre-downloaded HuggingFace datasets instead of the slow arXiv API.
Much faster and can easily scale to 1GB+ of data.

Available datasets:
1. 'scientific_papers' (arXiv + PubMed) - 200k+ papers, ~1.5GB
2. 'arxiv_dataset' - Full arXiv corpus, 1.7M+ papers, ~10GB+
3. 'pubmed' - PubMed abstracts, 5M+ papers, massive

Usage:
    python prepare_hf.py --dataset scientific_papers --num_samples 100000
    python prepare_hf.py --dataset arxiv_dataset --num_samples 500000
"""

import os
import pickle
import argparse
import numpy as np
import tiktoken
from tqdm import tqdm


def download_scientific_papers_hf(
    output_file: str = 'arxiv_abstracts.txt',
    num_samples: int = 100000,
    dataset_name: str = 'scientific_papers',
    subset: str = 'arxiv',
):
    """
    Download scientific papers using HuggingFace datasets (FAST!).

    Args:
        output_file: Where to save the downloaded text
        num_samples: Number of papers to download (default: 100k)
        dataset_name: HuggingFace dataset name
        subset: Dataset subset (e.g., 'arxiv', 'pubmed')

    Returns:
        True if successful, False otherwise
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("\nERROR: HuggingFace datasets library not installed!")
        print("Install with: pip install datasets")
        return False

    print(f"\n{'='*70}")
    print(f"Downloading {num_samples:,} papers from HuggingFace")
    print(f"Dataset: {dataset_name} ({subset})")
    print(f"{'='*70}\n")

    # Load dataset from HuggingFace
    print("Loading dataset from HuggingFace Hub...")
    try:
        if dataset_name == 'scientific_papers':
            # arXiv + PubMed scientific papers
            dataset = load_dataset('scientific_papers', subset, split='train')
        elif dataset_name == 'arxiv_dataset':
            # Full arXiv dataset (larger)
            dataset = load_dataset('arxiv_dataset', split='train')
        elif dataset_name == 'pubmed':
            # PubMed abstracts
            dataset = load_dataset('pubmed', split='train')
        else:
            print(f"Unknown dataset: {dataset_name}")
            return False
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("\nTrying alternative: ccdv/arxiv-summarization...")
        try:
            dataset = load_dataset('ccdv/arxiv-summarization', split='train')
        except Exception as e2:
            print(f"Error: {e2}")
            return False

    print(f"Dataset loaded! Total samples available: {len(dataset):,}")

    # Limit to requested number of samples
    num_samples = min(num_samples, len(dataset))
    dataset = dataset.select(range(num_samples))

    print(f"Processing {num_samples:,} samples...")

    # Extract text (handle different dataset formats)
    texts = []
    for sample in tqdm(dataset, desc="Extracting text"):
        text = ""

        # Try different field names (datasets have different schemas)
        if 'abstract' in sample:
            text = sample['abstract']
        elif 'article' in sample:
            text = sample['article']
        elif 'text' in sample:
            text = sample['text']
        elif 'summary' in sample:
            # Some datasets have summary + article
            if 'article' in sample:
                text = sample['article']
            else:
                text = sample['summary']

        # Clean and filter
        if isinstance(text, str) and len(text) > 100:
            # Remove extra whitespace
            clean_text = ' '.join(text.split())
            texts.append(clean_text)

    if not texts:
        print("\nERROR: No text extracted from dataset!")
        return False

    print(f"\nExtracted {len(texts):,} valid text samples")

    # Calculate size
    total_chars = sum(len(t) for t in texts)
    avg_chars = total_chars / len(texts)
    print(f"Total characters: {total_chars:,}")
    print(f"Average chars per sample: {avg_chars:.0f}")
    print(f"Estimated raw size: {total_chars / 1024**2:.1f} MB")

    # Write to file
    print(f"\nWriting to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n\n'.join(texts))

    print(f"Saved to {output_file}")

    # Print sample
    print("\n" + "="*70)
    print("Sample text (first entry):")
    print("="*70)
    print(texts[0][:500])
    if len(texts[0]) > 500:
        print("...")
    print("="*70)

    return True


def prepare_dataset(
    num_samples: int = 100000,
    dataset_name: str = 'scientific_papers',
    subset: str = 'arxiv',
):
    """
    Prepare scientific papers dataset using HuggingFace:
    1. Download papers from HuggingFace datasets
    2. Tokenize with GPT-2 BPE
    3. Create train/val split
    4. Save as binary files
    """

    # Download data
    input_file = os.path.join(os.path.dirname(__file__), 'arxiv_abstracts.txt')

    # Check if we need to download
    should_download = True
    if os.path.exists(input_file):
        file_size_mb = os.path.getsize(input_file) / 1024**2
        print(f"\nFound existing file: {input_file} ({file_size_mb:.1f} MB)")
        response = input("Download new data? (y/n): ").lower().strip()
        should_download = response == 'y'

    if should_download:
        success = download_scientific_papers_hf(
            input_file,
            num_samples=num_samples,
            dataset_name=dataset_name,
            subset=subset,
        )
        if not success:
            print("\nFailed to download dataset. Exiting.")
            return

    # Read the data
    with open(input_file, 'r', encoding='utf-8') as f:
        data = f.read()

    print(f"\nLoaded {len(data):,} characters of scientific text")
    print(f"File size: {len(data) / 1024**2:.1f} MB")

    # Tokenize with GPT-2 BPE (same tokenizer as TinyStories training)
    print("\nTokenizing with GPT-2 BPE...")
    enc = tiktoken.get_encoding("gpt2")
    tokens = enc.encode_ordinary(data)
    print(f"Total tokens: {len(tokens):,}")
    print(f"Tokenized size: {len(tokens) * 2 / 1024**2:.1f} MB (uint16)")

    # Vocab size
    vocab_size = enc.n_vocab
    print(f"Vocabulary size: {vocab_size:,}")

    # Train/val split (90/10)
    n = len(tokens)
    split_idx = int(0.9 * n)
    train_tokens = tokens[:split_idx]
    val_tokens = tokens[split_idx:]

    print(f"\nTrain tokens: {len(train_tokens):,} ({len(train_tokens) * 2 / 1024**2:.1f} MB)")
    print(f"Val tokens: {len(val_tokens):,} ({len(val_tokens) * 2 / 1024**2:.1f} MB)")

    # Save as binary files (uint16 since vocab_size < 65536)
    train_ids = np.array(train_tokens, dtype=np.uint16)
    val_ids = np.array(val_tokens, dtype=np.uint16)

    output_dir = os.path.dirname(__file__)
    train_file = os.path.join(output_dir, 'train.bin')
    val_file = os.path.join(output_dir, 'val.bin')

    print(f"\nSaving to binary files...")
    train_ids.tofile(train_file)
    val_ids.tofile(val_file)

    print(f"[OK] Saved train.bin: {len(train_ids):,} tokens")
    print(f"[OK] Saved val.bin: {len(val_ids):,} tokens")

    # Save metadata (for compatibility with training script)
    meta = {
        'vocab_size': vocab_size,
        'tokenizer': 'gpt2',
    }

    meta_file = os.path.join(output_dir, 'meta.pkl')
    with open(meta_file, 'wb') as f:
        pickle.dump(meta, f)

    print(f"[OK] Saved meta.pkl")

    print("\n" + "="*70)
    print("Dataset preparation complete!")
    print("="*70)
    print(f"Dataset: {dataset_name} ({subset})")
    print(f"Total samples: {num_samples:,}")
    print(f"Total tokens: {len(tokens):,}")
    print(f"Total size: {len(tokens) * 2 / 1024**2:.1f} MB")
    print("\nTo train, use: python train.py config/train_scientific.py")
    print("="*70)

    # Print sample tokens for verification
    print("\nSample decoded text from train set (first 200 tokens):")
    print("="*70)
    sample_text = enc.decode(train_tokens[:200])
    print(sample_text)
    print("="*70)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prepare scientific papers dataset from HuggingFace')
    parser.add_argument(
        '--num_samples',
        type=int,
        default=100000,
        help='Number of samples to download (default: 100k)',
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default='scientific_papers',
        choices=['scientific_papers', 'arxiv_dataset', 'pubmed'],
        help='HuggingFace dataset name',
    )
    parser.add_argument(
        '--subset',
        type=str,
        default='arxiv',
        choices=['arxiv', 'pubmed'],
        help='Dataset subset (for scientific_papers)',
    )

    args = parser.parse_args()

    print(f"\n{'='*70}")
    print("Scientific Papers Dataset Preparation (HuggingFace)")
    print(f"{'='*70}")
    print(f"Dataset: {args.dataset}")
    print(f"Subset: {args.subset}")
    print(f"Samples: {args.num_samples:,}")
    print(f"{'='*70}\n")

    prepare_dataset(
        num_samples=args.num_samples,
        dataset_name=args.dataset,
        subset=args.subset,
    )
