import argparse
import json
from diac_btc.text import MERGED_DIAC_TO_TOKEN_MAP
import re
import os
from collections import Counter

chars_to_ignore_regex = r'[,\?\.\!\-\;\:\"\“%\‘\”�…{}\【\】・。『』、ー〜\ufeff\[\]]'  # remove special character tokens

def build_charset(manifest_paths, extra_chars=None):
    char_counts = Counter()
    for mf in manifest_paths:
        with open(mf, "r", encoding="utf-8") as f:
            for line in f:
                t = json.loads(line)["text"]
                char_counts.update(list(t))
    
    if extra_chars:
        for char in extra_chars:
            char_counts[char] += 1

    # Remove newline/tab just in case
    char_counts.pop("\n", None)
    char_counts.pop("\t", None)

    chars = sorted(char_counts.keys())
    return chars, char_counts

def remove_special_characters(data):
    data["text"] = re.sub(chars_to_ignore_regex, "", data["text"])
    # remove zero width spaces
    data["text"] = data["text"].replace("\u200b", "")
    return data

def replace_merged_diacritics(data):
    for k, v in MERGED_DIAC_TO_TOKEN_MAP.items():
        data["text"] = data["text"].replace(k, v)
    return data

def normalize_english_chars(data):
    data["text"] = data["text"].replace("|", " ")
    data["text"] = data["text"].replace("-", " ")
    # convert to lowercase
    data["text"] = data["text"].lower()
    return data

def preprocess_manifest(manifest_paths):
    processed_manifest_paths = []
    for mf in manifest_paths:
        processed_mf = mf.replace(".json", "_preprocessed.json")
        with open(mf, "r", encoding="utf-8") as f, open(processed_mf, "w", encoding="utf-8") as f_out:
            for line in f:
                item = json.loads(line)
                item = remove_special_characters(item)
                item = replace_merged_diacritics(item)
                item = normalize_english_chars(item)
                f_out.write(json.dumps(item, ensure_ascii=False) + "\n")
            processed_manifest_paths.append(processed_mf)
    return processed_manifest_paths

def main(args):

    processed_manifest_paths = preprocess_manifest(args.input_paths)
    chars, char_counts = build_charset(processed_manifest_paths, args.extra_chars)
    print(f"Char set: {chars}")
    
    if args.split != "test":
        char_file_path = os.path.join(args.output_path, "charset.txt")
        with open(char_file_path, "w", encoding="utf-8") as f:
            for c in chars:
                f.write(c + "\n")
        
        # Dump character counts
        char_counts_path = os.path.join(args.output_path, "charset_counts.txt")
        # sort and dump the character counts
        char_counts = sorted(char_counts.items(), key=lambda x: x[1], reverse=True)
        with open(char_counts_path, "w", encoding="utf-8") as f:
            for c, count in char_counts:
                f.write(f"{c}\t{count}\n")

    # combine and dump the processed manifest paths to a single file
    combined_manifest_path = os.path.join(args.output_path, f"{args.split}_manifest.json")
    with open(combined_manifest_path, "w", encoding="utf-8") as f:
        for mf in processed_manifest_paths:
            with open(mf, "r", encoding="utf-8") as f_in:
                for line in f_in:
                    f.write(line)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_paths", "-i",
        nargs='+',
        required=True,
        help="List of manifest file paths."
    )
    parser.add_argument(
        "--split", "-s",
        default="combined",
        help="Split of the dataset."
    )
    parser.add_argument(
        "--extra_chars", "-e",
        nargs='+',
        required=False,
        help="List of extra characters to add to the charset."
    )
    parser.add_argument(
        "--output_path", "-o",
        default="outputs/nemo",
        help="Path to save the charset file."
    )
    args = parser.parse_args()

    main(args)