import argparse
import os
import pandas as pd
from diac_btc.models import Wav2Vec2DiacritizationModel
from tqdm import tqdm

def main():
    model = Wav2Vec2DiacritizationModel(args.model_path)

    if os.path.isfile(args.text):
        # expect a tsv file with columns: audio_path, text
        df = pd.read_csv(args.text, sep="\t")
        with open(args.output_path, "w") as f:
            for index, row in tqdm(df.iterrows()):
                audio_path = row["audio_path"]
                text = row["text"]
                diacritized_text = model.diacritize(text, audio_path, args.constrained, args.method)
                f.write(f"{diacritized_text}\n")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="jonatasgrosman/wav2vec2-large-xlsr-53-arabic")
    parser.add_argument("--text", type=str, required=True)
    parser.add_argument("--constrained", type=bool, default=True)
    parser.add_argument("--method", type=str, default="wfst")
    parser.add_argument("--output_path", type=str, required=True)
    args = parser.parse_args()

    main(args)