import argparse
import json
import random
import os
from pathlib import Path

'''
Script to split a train manifest into train and validation sets.
Usage: 
$ python scripts/nemo/split_train_val.py --input_manifest outputs/nemo/combined_manifest.json \
    --train_output outputs/nemo/train_manifest.json \
    --val_output outputs/nemo/val_manifest.json \
    --val_ratio 0.1 \
    --no_shuffle \
    --seed 42
'''


def split_manifest(input_manifest, train_output, val_output, val_ratio=0.1, shuffle=True, seed=42):
    """
    Split a manifest file into train and validation sets.
    
    Args:
        input_manifest: Path to input manifest file (JSONL format)
        train_output: Path to output train manifest file
        val_output: Path to output validation manifest file
        val_ratio: Ratio of data to use for validation (default: 0.1)
        shuffle: Whether to shuffle the data before splitting (default: True)
        seed: Random seed for reproducibility (default: 42)
    """
    # Read all lines from the manifest
    items = []
    with open(input_manifest, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    
    print(f"Total items in manifest: {len(items)}")
    
    # Shuffle if requested
    if shuffle:
        random.seed(seed)
        random.shuffle(items)
    
    # Calculate split point
    val_size = int(len(items) * val_ratio)
    train_size = len(items) - val_size
    
    # Split the data
    train_items = items[:train_size]
    val_items = items[train_size:]
    
    print(f"Train items: {len(train_items)}")
    print(f"Validation items: {len(val_items)}")
    
    # Write train manifest
    os.makedirs(os.path.dirname(train_output) if os.path.dirname(train_output) else ".", exist_ok=True)
    with open(train_output, "w", encoding="utf-8") as f:
        for item in train_items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    # Write validation manifest
    os.makedirs(os.path.dirname(val_output) if os.path.dirname(val_output) else ".", exist_ok=True)
    with open(val_output, "w", encoding="utf-8") as f:
        for item in val_items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    print(f"Train manifest saved to: {train_output}")
    print(f"Validation manifest saved to: {val_output}")


def main():
    parser = argparse.ArgumentParser(
        description="Split a train manifest into train and validation sets"
    )
    parser.add_argument(
        "--input_manifest", "-i",
        type=str,
        required=True,
        help="Path to input train manifest file (JSONL format)"
    )
    parser.add_argument(
        "--train_output", "-t",
        type=str,
        required=True,
        help="Path to output train manifest file"
    )
    parser.add_argument(
        "--val_output", "-v",
        type=str,
        required=True,
        help="Path to output validation manifest file"
    )
    parser.add_argument(
        "--val_ratio", "-r",
        type=float,
        default=0.1,
        help="Ratio of data to use for validation (default: 0.1, i.e., 10%%)"
    )
    parser.add_argument(
        "--no_shuffle",
        action="store_true",
        help="Don't shuffle the data before splitting"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    
    args = parser.parse_args()
    
    # Validate val_ratio
    if not 0 < args.val_ratio < 1:
        raise ValueError("val_ratio must be between 0 and 1")
    
    split_manifest(
        input_manifest=args.input_manifest,
        train_output=args.train_output,
        val_output=args.val_output,
        val_ratio=args.val_ratio,
        shuffle=not args.no_shuffle,
        seed=args.seed
    )


if __name__ == "__main__":
    main()
