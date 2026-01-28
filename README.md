# Automatic Diacritization

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rufaelfekadu/diac-btc/blob/main/notebooks/test.ipynb)

## Getting started

### Installation

1. Create and activate a virtual environment:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

2. Install the package:

   ```bash
   pip install -e .
   ```

3. Install k2 (required for CTC decoding):

   ```bash
   # For CUDA 12.8 with PyTorch 2.9.1
   pip install k2==1.24.4.dev20251118+cuda12.8.torch2.9.1 -f https://k2-fsa.github.io/k2/cuda.html

   # For other CUDA/PyTorch versions, check: https://k2-fsa.github.io/k2/installation/index.html
   ```

**Note:** k2 requires CUDA and must match your PyTorch and CUDA versions. Visit the [k2 installation guide](https://k2-fsa.github.io/k2/installation/index.html) to find the correct version for your setup.

## Inference

Run inference on audio files using trained models. The script supports both Wav2Vec2 and NeMo models.

### Basic Usage

```bash
python inference.py \
    --model_type wav2vec2 \
    --model_path jonatasgrosman/wav2vec2-large-xlsr-53-arabic \
    --manifest_paths data/clartts/raw/test/metadata.json \
    --constrained True \
    --method wfst \
    --output_path outputs/
```

### Parameters

- `--model_type`: Model type to use (`wav2vec2` or `nemo`)
- `--model_path`: Path to the model (HuggingFace model ID for wav2vec2, or path to `.nemo` file for NeMo)
- `--manifest_paths`: One or more manifest file paths (JSON or TSV format)
- `--constrained`: Whether to use constrained decoding (default: `True`)
- `--method`: Decoding method (`wfst` or `ctc`)
- `--output_path`: Base output directory (default: `outputs/`)

### Output

The script generates:
- `pred.txt`: Predicted diacritized text (one line per input)
- `gt.txt`: Ground truth diacritized text (one line per input)
- `inference.log`: Inference log file with timing information

Outputs are saved to: `{output_path}/{model_type}/{method}/{constrained}/{dataset_name}/`

## Evaluation

Evaluate model predictions using DER (Diacritic Error Rate), WER (Word Error Rate), and SER (Sentence Error Rate).

### Basic Usage

```bash
python eval.py \
    -ofp outputs/wav2vec2/wfst/constrained/clartts/gt.txt \
    -tfp outputs/wav2vec2/wfst/constrained/clartts/pred.txt \
    --log_file outputs/wav2vec2/wfst/constrained/clartts/eval.log
```

### Parameters

- `-ofp, --original-file-path`: Path to ground truth file (required)
- `-tfp, --target-file-path`: Path to predictions file (required)
- `-s, --style`: Evaluation style (`Fadel` or `Zitouni`, default: `Fadel`)
- `--log_file`: Path to log file (default: `eval.log`)

### Evaluation Metrics

The script calculates DER, WER, and SER with the following configurations:
- With/without case ending
- Including/excluding no diacritic cases

Results are displayed in formatted tables and saved to the log file.

## Batch Evaluation

Run inference and evaluation for multiple datasets, models, and methods:

```bash
bash scripts/eval.sh
```

Edit `scripts/eval.sh` to configure:
- Datasets to evaluate
- Models to use
- Decoding methods (wfst, ctc)

## Training

See individual README files for training workflows:
- [NeMo Fine-tuning](scripts/nemo/README.md)
- [Wav2Vec2 Training](scripts/wav2vec/README.md)
