import argparse
import json
from diac_btc.text import MERGED_DIAC_TO_TOKEN_MAP

def build_charset(manifest_paths, extra_chars=None):
    chars = set()
    for mf in manifest_paths:
        with open(mf, "r", encoding="utf-8") as f:
            for line in f:
                t = json.loads(line)["text"]
                chars.update(list(t))
    if extra_chars:
        chars.update(extra_chars)

    # Remove newline/tab just in case
    chars.discard("\n"); chars.discard("\t")

    return sorted(chars)

def preprocess_manifest(manifest_paths):
    for mf in manifest_paths:
        with open(mf, "r", encoding="utf-8") as f, open(mf.replace(".json", "_preprocessed.json"), "w", encoding="utf-8") as f_out:
            for line in f:
                item = json.loads(line)
                for k, v in MERGED_DIAC_TO_TOKEN_MAP.items():
                    item["text"] = item["text"].replace(k, v)
                temp = item['text']
                # item["text"] = item["text"].translate(str.maketrans(MERGED_DIAC_TO_TOKEN_MAP))
                json.dump(item, f_out, ensure_ascii=False)
                f_out.write("\n")
    return 

def main(args):

    chars = build_charset(args.input_paths, args.extra_chars)
    preprocess_manifest(args.input_paths)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_paths", "-i",
        nargs='+',
        required=True,
        help="List of manifest file paths."
    )
    parser.add_argument(
        "--extra_chars", "-e",
        nargs='+',
        required=False,
        help="List of extra characters to add to the charset."
    )
    parser.add_argument(
        "--output_path", "-o",
        default="outputs/charset.txt",
        help="Path to save the charset file."
    )
    args = parser.parse_args()

    main(args)