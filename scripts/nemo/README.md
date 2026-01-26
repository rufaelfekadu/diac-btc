# NeMo Fine-tuning

Scripts for fine-tuning NVIDIA NeMo ASR models by converting BPE-based models to character-based CTC models.

## Workflow

### 1. Prepare Manifests

Preprocess manifest files and build character sets:

```bash
python scripts/nemo/nemo_prep_manifest.py \
    --input_paths path/to/manifest1.json path/to/manifest2.json \
    --split combined \
    --output_path outputs/nemo
```

### 2. Split Train/Validation

Split manifest into train and validation sets:

```bash
python scripts/nemo/split_train_val.py \
    --input_manifest outputs/nemo/combined_manifest.json \
    --train_output outputs/nemo/train_manifest.json \
    --val_output outputs/nemo/val_manifest.json \
    --val_ratio 0.1
```

### 3. Fine-tune Model

Fine-tune pretrained BPE model as character-based CTC:

```bash
python scripts/nemo/nemo_speech_to_text_ctc.py \
    --config-path conf \
    --config-name conformer_ctc_char \
    model.train_ds.manifest_filepath=outputs/nemo/train_manifest.json \
    model.validation_ds.manifest_filepath=outputs/nemo/val_manifest.json \
    init_from_nemo_model=path/to/pretrained_model.nemo \
    trainer.devices=1 \
    trainer.max_epochs=50
```

## Configuration

Edit `conf/conformer_ctc_char.yaml` or override parameters via command line:
- `init_from_nemo_model`: Path to pretrained BPE `.nemo` model
- `freeze_encoder`: Freeze encoder weights (default: true)
- `model.labels`: Character vocabulary list
- `trainer.devices`: Number of GPUs (-1 for all)

## Utilities

Convert hybrid models to CTC:

```bash
python scripts/nemo/nemo_convert_to_ctc.py \
    --input path/to/hybrid_model.nemo \
    --output path/to/ctc_model.nemo \
    --model_type ctc
```
