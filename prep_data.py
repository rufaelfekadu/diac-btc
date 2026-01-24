# data preparation script for ClarTTS and ArVoice datasets, clartts is on the web at MBZUAI/clartts and arvoice is availabel locally at/l/ArVoice/v1, I want to merge the datasets and save a huggingface dataset 

from datasets import load_dataset, concatenate_datasets, DatasetDict, load_from_disk
from datasets import Audio
from transformers import AutoProcessor, AutoModelForCTC
import librosa
import torch
import numpy as np
import soundfile
import os
from functools import partial
import json
from tqdm import tqdm

# wav2vec_processor = AutoProcessor.from_pretrained("jonatasgrosman/wav2vec2-large-xlsr-53-arabic")
# wav2vec_model = AutoModelForCTC.from_pretrained("jonatasgrosman/wav2vec2-large-xlsr-53-arabic")

clartts_configs = {
    "hf_dataset": "MBZUAI/clartts",
    "output_dir": "data/clartts/raw",
    "splits": ["train", "test"],
    "num_proc": 16,
}

arvoice_configs = {
    "hf_dataset": "csv",
    "base_dir": "/l/ArVoice/v1",
    "csv_files": [
        "/l/ArVoice/v1/part-1/metadata_{}.csv",
        "/l/ArVoice/v1/part-2/metadata_{}.csv",
        ],
    "output_dir": "data/arvoice/raw",
    "num_proc": 16,
    "splits": ["train", "test"],
}
nadi_configs = {
    "hf_dataset": "MBZUAI/NADI-2025-Sub-task-3-test",
    "output_dir": "data/nadi/raw",
    "splits": ["test"],
    "num_proc": 16,
}


def process_clartts(item, split):
    # audio is a dict with 'array' and 'sampling_rate' if loaded as Audio, else just a list of floats
    audio = item['audio']
    sr = item['sampling_rate']

    # If 'audio' is a dict with 'array', get the array, else assume it's already a list
    if isinstance(audio, dict) and 'array' in audio:
        audio_array = np.array(audio['array'], dtype=np.float32)
    else:
        audio_array = np.array(audio, dtype=np.float32)

    # Resample if necessary
    if sr != 16000:
        audio_array = librosa.resample(audio_array, orig_sr=sr, target_sr=16000)
        sr = 16000

    # save audio to disk
    audio_path = item['file']
    audio_path = os.path.join(clartts_configs['splits'][split]['output_dir'], audio_path)
    os.makedirs(os.path.dirname(audio_path), exist_ok=True)

    item['audio_filepath'] = audio_path
    item['sampling_rate'] = sr
    item['duration'] = librosa.get_duration(y=audio_array, sr=sr)

    if os.path.isfile(audio_path):
        return item
    soundfile.write(audio_path, audio_array, sr, format='wav')
    
    return item

def process_arvoice(item, split):

    audio, sr = librosa.load(item['file_name'])
    duration = librosa.get_duration(y=audio, sr=sr)

    # resample if necessary and save to output dir
    if sr != 16000:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        sr = 16000

    audio_path = item['file_name']
    # Maintain folder structure after arvoice_configs['base_dir']
    rel_path = os.path.relpath(audio_path, arvoice_configs['base_dir'])
    audio_path = os.path.join(arvoice_configs['output_dir'], split, rel_path)
    os.makedirs(os.path.dirname(audio_path), exist_ok=True)

    item['duration'] = duration
    item['sampling_rate'] = sr
    item['audio_filepath'] = audio_path

    if os.path.isfile(audio_path):
        return item

    soundfile.write(audio_path, audio, sr, format='wav')

    return item

def process_nadi(item, split):

    audio = item['audio']['array']
    audio_array = np.array(audio, dtype=np.float32)
    sr = item['audio']['sampling_rate']
    if sr != 16000:
        audio_array = librosa.resample(audio_array, orig_sr=sr, target_sr=16000)
        sr = 16000
    # get unique id for the audio
    unique_id = str(item['id'])
    audio_path = os.path.join(nadi_configs['output_dir'], split, unique_id + ".wav")
    os.makedirs(os.path.dirname(audio_path), exist_ok=True)

    item['audio_filepath'] = audio_path
    item['sampling_rate'] = sr
    item['duration'] = librosa.get_duration(y=audio_array, sr=sr)

    if os.path.isfile(audio_path):
        return item
    
    soundfile.write(audio_path, audio_array, sr, format='wav')

    return item

def main():

    #CLArTTS
    for split in clartts_configs['splits']:
        clartts = load_dataset("MBZUAI/clartts", split=split)
        process_func = partial(process_clartts, split=split)
        clartts_dataset_processed = clartts.map(process_func, num_proc=16)

        # dump the metadata to json
        data_itr = iter(clartts_dataset_processed)
        with open(os.path.join(clartts_configs['output_dir'], split, "metadata.json"), "w") as f:
            for item in tqdm(data_itr):
                del item['audio']
                json.dump(item, f, ensure_ascii=False, indent=4)
                f.write("\n")

    # ArVoice
    for split in arvoice_configs['splits']:
        total_arvoice = []
        prepocess_arvoice_fn = partial(process_arvoice, split=split)
        for csv_file in arvoice_configs['csv_files']:
            arvoice = load_dataset("csv", data_files=csv_file.format(split))["train"]
            arvoice = arvoice.map(lambda x:{"source": x["file_name"].split("/")[-3]}, num_proc=arvoice_configs['num_proc'])
            arvoice = arvoice.map(prepocess_arvoice_fn, num_proc=arvoice_configs['num_proc'])
            total_arvoice.append(arvoice)
        total_arvoice = concatenate_datasets(total_arvoice)

        total_arvoice = total_arvoice.rename_column("transcription", "text")

        data_itr = iter(total_arvoice)
        with open(os.path.join(arvoice_configs['output_dir'], split, "metadata.json"), "w") as f:
            for item in tqdm(data_itr):
                if 'audio' in item:
                    del item['audio']
                json.dump(item, f, ensure_ascii=False, indent=4)
                f.write("\n")

    #NADI
    for split in nadi_configs['splits']:
        process_nadi_fn = partial(process_nadi, split=split)
        nadi = load_dataset("MBZUAI/NADI-2025-Sub-task-3-test", split=split)
        nadi = nadi.map(lambda example, idx: {"id": idx}, with_indices=True, num_proc=nadi_configs['num_proc'])
        nadi = nadi.map(process_nadi_fn, num_proc=nadi_configs['num_proc'])

        nadi = nadi.rename_column("transcription", "text")

        # dump the metadata to json
        data_itr = iter(nadi)
        with open(os.path.join(nadi_configs['output_dir'], split, "metadata.json"), "w") as f:
            for item in tqdm(data_itr):
                del item['audio']
                json.dump(item, f, ensure_ascii=False, indent=4)
                f.write("\n")


if __name__ == "__main__":
    main()



