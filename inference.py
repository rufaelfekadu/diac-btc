import argparse
import os
import pandas as pd
from diac_btc.models import Wav2Vec2DiacritizationModel
from tqdm import tqdm
import json


def main(args):

    model = Wav2Vec2DiacritizationModel(args.model_path)

    _, ext = os.path.splitext(args.text)
    ext = ext.lower()
    if ext == ".tsv":
        # expect a tsv file with columns: audio_path, text
        df = pd.read_csv(args.text, sep="\t")
        gt_path = os.path.join(args.output_path, "gt.txt")
        pred_path = os.path.join(args.output_path, "pred.txt")
        with open(pred_path, "w") as pred_f, open(gt_path, "w") as gt_f:
            for index, row in tqdm(df.iterrows(), total=df.shape[0]):
                audio_path = row["audio_path"]
                text = row["text"]
                diacritized_text = model.diacritize(text, audio_path, args.constrained, args.method)
                pred_f.write(f"{diacritized_text}\n")
                gt_f.write(f"{text}\n")

    elif ext == ".json":
        with open(args.text, "r") as infile:
            content = infile.read().strip()
            try:
                data = json.loads(content)
                if isinstance(data, dict):
                    # single dict, wrap as list
                    data = [data]
            except Exception:
                # fallback to line-delimited JSON
                data = []
                with open(args.text, "r") as inf:
                    for line in inf:
                        line = line.strip()
                        if line:
                            data.append(json.loads(line))

        # Assume gt.txt contains 1 raw text per line in same order
        gt_path = os.path.join(args.output_path, "gt.txt")
        pred_path = os.path.join(args.output_path, "pred.txt")  
        with open(pred_path, "w") as pred_f, open(gt_path, "w") as gt_f:
            for idx, entry in enumerate(tqdm(data, total=len(data))):
                audio_path = entry.get("audio_filepath")
                text = entry.get("text")
                diacritized_text = model.diacritize(text, audio_path, args.constrained, args.method)
                pred_f.write(f"{diacritized_text}\n")
                gt_f.write(f"{text}\n")

    else:
        raise ValueError("Unsupported input file format. Only .tsv and .json are supported.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="jonatasgrosman/wav2vec2-large-xlsr-53-arabic")
    parser.add_argument("--text", type=str, default="data/clartts/raw/test/metadata.json")
    parser.add_argument("--constrained", type=bool, default=True)
    parser.add_argument("--method", type=str, default="wfst")
    parser.add_argument("--output_path", type=str, default="outputs/wav2vec/clartts/wfst")
    args = parser.parse_args()

    os.makedirs(args.output_path, exist_ok=True)
    main(args)