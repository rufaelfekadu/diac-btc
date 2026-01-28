# inference
declare -A DATASETS
DATASETS=(
    ["clartts"]="data/clartts/raw/test/metadata.json"
    # ["arvoice"]="data/arvoice/raw/test/metadata.json"
    # ["tuneSwitch"]="data/tuneSwitch/raw/validation/metadata.json"
    # ["mixat"]="data/mixat/raw/test/metadata.json"
)
declare -A MODELS
MODELS=(
    ["wav2vec2"]="jonatasgrosman/wav2vec2-large-xlsr-53-arabic"
    # ["nemo"]="outputs/nemo/ckpts/stt_ar_fastconformer_ctc_large_pcd.v1.0.nemo"
)
declare -A METHODS
METHODS=(
    ["wfst"]="wfst"
    ["ctc"]="ctc"
)

for dataset_key in "${!DATASETS[@]}"; do
    for model_key in "${!MODELS[@]}"; do
        for method_key in "${!METHODS[@]}"; do
            dataset=${DATASETS[$dataset_key]}
            model=${MODELS[$model_key]}
            method=${METHODS[$method_key]}
            echo "Inference ${dataset_key} with ${model_key} and ${method_key}"
            python inference.py \
                --model_type ${model_key} \
                --model_path ${model} \
                --manifest_paths ${dataset} \
                --constrained True \
                --method ${method} \
                --output_path outputs/
            
            python eval.py \
                -ofp  outputs/${model_key}/${method_key}/${dataset_key}/gt.txt \
                -tfp  outputs/${model_key}/${method_key}/${dataset_key}/pred.txt \
                --log_file outputs/${model_key}/${method_key}/${dataset_key}/eval.log
                
        done
    done
done

