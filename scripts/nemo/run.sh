#!/bin/bash
set -e          # Exit immediately if a command exits with a non-zero status
set -o pipefail # Return value of a pipeline is the status of the last command to exit with a non-zero status
set -u          # Exit on undefined variables

# Error handler
trap 'echo "Error: Script failed at line $LINENO. Command: $BASH_COMMAND"' ERR

################################# Download and Convert Model #################################
# download nemo model from hf
# hf_model_id="nvidia/stt_ar_fastconformer_hybrid_large_pcd_v1.0"
# hf_output_path="outputs/nemo/ckpts/stt_ar_fastconformer_hybrid_large_pcd.v1.0.nemo"
# if [ ! -f $hf_output_path ]; then
#     hf download $hf_model_id stt_ar_fastconformer_hybrid_large_pcd_v1.0.nemo --local-dir outputs/nemo/ckpts --cache-dir outputs/nemo/ckpts/.cache
#     rm -rf outputs/nemo/ckpts/.cache
# fi

# # convert model to ctc
# python scripts/nemo/convert_hybrid_to_ctc.py \
#     -i outputs/nemo/ckpts/stt_ar_fastconformer_hybrid_large_pcd_v1.0.nemo \
#     -o outputs/nemo/ckpts/stt_ar_fastconformer_ctc_large_pcd_v1.0.nemo \
#     --model_type ctc

# ################################# Data Preparation #################################
# prepare train manifest
python scripts/nemo/prep_manifest.py \
    --input_paths \
        data/clartts/raw/train/clartts_train_metadata.json \
        data/arvoice/raw/train/arvoice_train_metadata.json \
    --split combined \
    --output_path outputs/nemo

# prepare test manifest
python scripts/nemo/prep_manifest.py \
    --input_paths \
        data/clartts/raw/test/clartts_test_metadata.json \
        data/arvoice/raw/test/arvoice_test_metadata.json \
    --split test \
    --output_path outputs/nemo

# Train Tokenizer
python scripts/nemo/process_asr_text_tokenizer.py \
    --manifest=outputs/nemo/combined_manifest.json \
    --data_root=outputs/nemo \
    --vocab_size=1024 \
    --tokenizer=spe \
    --spe_type=char \
    --log

# split train and val
python scripts/nemo/split_train_val.py \
    --input_manifest outputs/nemo/combined_manifest.json \
    --train_output outputs/nemo/train_manifest.json \
    --val_output outputs/nemo/val_manifest.json \
    --val_ratio 0.1

# # ################################# Model Finetuning #################################
# # finetune model
python scripts/nemo/speech_to_text_ctc_bpe.py \
    --config-path conf/fast-conformer_ctc_bpe.yaml \
    --config-name fast-conformer_ctc_bpe \
    init_from_nemo_model=outputs/nemo/ckpts/stt_ar_fastconformer_ctc_large_pcd.v1.0.nemo \
    model.train_ds.manifest_filepath=outputs/nemo/train_manifest_temp.json \
    model.validation_ds.manifest_filepath=outputs/nemo/val_manifest_temp.json \
    model.tokenizer.dir=outputs/nemo/tokenizer_spe_char_v1024 \
    model.tokenizer.type=bpe


