import argparse
import os
import pandas as pd
from diac_btc.models import Wav2Vec2DiacritizationModel, NemoDiacritizationModel
from tqdm import tqdm
import json
import time
from pyarabic import araby


def main(args):

    if args.model_type == "wav2vec2":
        model = Wav2Vec2DiacritizationModel(args.model_path)
    elif args.model_type == "nemo":
        model = NemoDiacritizationModel(args.model_path)
    else:
        raise ValueError("Unsupported model type. Only wav2vec2 and nemo are supported.")

    _, ext = os.path.splitext(args.text)
    ext = ext.lower()

    if ext == ".tsv":
        df = pd.read_csv(args.text, sep="\t")
        
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
        df = pd.DataFrame(data)
        
    else:
        raise ValueError("Unsupported input file format. Only .tsv and .json are supported.")

    # Assume gt.txt contains 1 raw text per line in same order
    gt_path = os.path.join(args.output_path, "gt.txt")
    pred_path = os.path.join(args.output_path, "pred.txt")  
    rtf_times = []
    with open(pred_path, "w") as pred_f, open(gt_path, "w") as gt_f:
        for idx, entry in tqdm(df.iterrows(), total=df.shape[0]):
            audio_path = entry["audio_filepath"]
            text = entry["text"]
            text_no_diac = araby.strip_diacritics(text)
            diacritized_text, rtf = model.diacritize(text_no_diac, audio_path, args.constrained, args.method)
            if len(diacritized_text) == 0:
                print(f"WARNING: Empty diacritized text for {audio_path}")
                diacritized_text = text_no_diac
            rtf_times.append(rtf)
            pred_f.write(f"{diacritized_text}\n")
            gt_f.write(f"{text}\n")

    print(f"Average rtfx time: {1/(sum(rtf_times) / len(rtf_times))} seconds")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, choices=["wav2vec2", "nemo"], default="wav2vec2")
    parser.add_argument("--model_path", type=str, default="jonatasgrosman/wav2vec2-large-xlsr-53-arabic")
    parser.add_argument("--text", type=str, default="data/clartts/raw/test/metadata.json")
    parser.add_argument("--constrained", type=bool, default=True)
    parser.add_argument("--method", type=str, default="wfst")
    parser.add_argument("--output_path", type=str, default="outputs/wav2vec/clartts/wfst")
    args = parser.parse_args()

    os.makedirs(args.output_path, exist_ok=True)
    main(args)