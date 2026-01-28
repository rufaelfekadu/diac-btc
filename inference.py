import argparse
import os
import pandas as pd
from diac_btc.models import Wav2Vec2DiacritizationModel, NemoDiacritizationModel
from tqdm import tqdm
import json
import time
from pyarabic import araby
import logging
from diac_btc.text import preprocess_text
# setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def pred_manifest(model, text, output_path):

    # read manifet and write the predictions
    base_name = os.path.basename(text)
    name, ext = os.path.splitext(base_name)
    output_path = os.path.join(output_path, name)
    os.makedirs(output_path, exist_ok=True)

    if ext == ".tsv":
        df = pd.read_csv(text, sep="\t")
    elif ext == ".json":
        with open(text, "r") as infile:
            content = infile.read().strip()
            try:
                data = json.loads(content)
                if isinstance(data, dict):
                    # single dict, wrap as list
                    data = [data]
            except Exception:
                # fallback to line-delimited JSON
                data = []
                with open(text, "r") as inf:
                    for line in inf:
                        line = line.strip()
                        if line:
                            data.append(json.loads(line))
        df = pd.DataFrame(data)
    else:
        raise ValueError("Unsupported input file format. Only .tsv and .json are supported.")
    
    gt_path = os.path.join(output_path, "gt.txt")
    pred_path = os.path.join(output_path, "pred.txt") 
    rtf_times = []
    with open(pred_path, "w") as pred_f, open(gt_path, "w") as gt_f:
        for idx, entry in tqdm(df.iterrows(), total=df.shape[0]):
            audio_path = entry["audio_filepath"]
            text = entry["text"]
            text_no_diac = araby.strip_diacritics(text)
            text_no_diac = preprocess_text(text_no_diac)
            diacritized_text, rtf = model.diacritize(text_no_diac, audio_path, args.constrained, args.method)
            if len(diacritized_text) != 0 and text_no_diac != "":
                pred_f.write(f"{diacritized_text}\n")
                gt_f.write(f"{text}\n")
                rtf_times.append(rtf)
            else:
                logger.warning(f"Empty diacritized text for {audio_path}: {text}")
                diacritized_text = text_no_diac
                rtf_times.append(rtf)


    logger.info(f"Average rtfx time: {1/(sum(rtf_times) / len(rtf_times))} seconds")

def main(args):

    if args.model_type == "wav2vec2":
        model = Wav2Vec2DiacritizationModel(args.model_path)
    elif args.model_type == "nemo":
        model = NemoDiacritizationModel(args.model_path)
    else:
        raise ValueError("Unsupported model type. Only wav2vec2 and nemo are supported.")

    for manifest in args.manifest_paths:
        logger.info(f"Processing {manifest}")
        pred_manifest(model, manifest, args.output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, choices=["wav2vec2", "nemo"], default="wav2vec2")
    parser.add_argument("--model_path", type=str, default="jonatasgrosman/wav2vec2-large-xlsr-53-arabic")
    parser.add_argument("--manifest_paths", nargs="+", type=str, default=["data/clartts/raw/test/clartts_test_metadata.json"])
    parser.add_argument("--constrained", type=bool, default=True)
    parser.add_argument("--method", type=str, default="wfst")
    parser.add_argument("--output_path", type=str, default="outputs")
    args = parser.parse_args()
 
    constrained = "constrained" if args.constrained else "unconstrained"
    args.output_path = os.path.join(args.output_path, args.model_type, args.method, constrained)

    os.makedirs(args.output_path, exist_ok=True)
    # add file handler to the logger
    file_handler = logging.FileHandler(os.path.join(args.output_path, "inference.log"))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(file_handler)
    
    main(args)