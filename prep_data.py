# data preparation script for ClarTTS and ArVoice datasets, clartts is on the web at MBZUAI/clartts and arvoice is availabel locally at/l/ArVoice/v1, I want to merge the datasets and save a huggingface dataset 

from datasets import load_dataset, concatenate_datasets
from datasets import Audio
import librosa
import numpy as np
import soundfile
import os
from functools import partial
import json
from tqdm import tqdm
import argparse
import shutil


TARGET_SR = 16000
NUM_PROC = 16

clartts_configs = {
    "name": "clartts",
    "hf_dataset": "MBZUAI/clartts",
    "output_dir": os.path.abspath("data/clartts/raw"),
    "splits": ["train", "test"],
    "audio_col": "audio",
    "text_col": "text",
    "filename_col": "file",
    "sampling_rate": 16000,
}

fleurs_configs = {
    "name": "fleurs",
    "hf_dataset": "google/fleurs",
    "output_dir": os.path.abspath("data/fleur/raw"),
    "splits": ["train", "valid", "test"],
    "audio_col": "audio_path",
    "text_col": "transcription",
    "filename_col": "audio_path",
    "sampling_rate": 16000,
}

nadi_configs = {
    "name": "nadi",
    "hf_dataset": "MBZUAI/NADI-2025-Sub-task-3-test",
    "output_dir": os.path.abspath("data/nadi/raw"),
    "splits": ["test"],
    "audio_col": "audio",
    "text_col": "transcription",
    "filename_col": "id",
    "sampling_rate": 16000,
}

tuneSwitch_configs = {
    "name": "tuneSwitch",
    "hf_dataset": "MBZUAI/TunSwitch",
    "output_dir": os.path.abspath("data/tuneSwitch/raw"),
    "splits": ["train", "validation"],
    "audio_col": "file_name",
    "text_col": "diacritized_transcription",
    "filename_col": None,
    "sampling_rate": 16000,
}

mixat_configs = {
    "name": "mixat",
    "hf_dataset": "MBZUAI/MIXAT",
    "output_dir": os.path.abspath("data/mixat/raw"),
    "splits": ["train"],
    "audio_col": "audio",
    "text_col": "transcript",
    "filename_col": None,
    "sampling_rate": 16000,
}

arvoice_configs = {
    "name": "arvoice",
    "hf_dataset": "csv",
    "base_dir": "/l/ArVoice/v1",
    "csv_files": [
        "/l/ArVoice/v1/part-1/metadata_{}.csv",
        "/l/ArVoice/v1/part-2/metadata_{}.csv",
        ],
    "output_dir": os.path.abspath("data/arvoice/raw"),
    "splits": ["train", "test"],
    "audio_col": "file_name",
    "text_col": "transcription",
    "filename_col": "file_name",
    "sampling_rate": 16000,
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
    audio_path = os.path.join(clartts_configs['output_dir'], split, audio_path)
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

def process_fleurs(item, split):
    audio = item['audio']['array']
    audio_array = np.array(audio, dtype=np.float32)
    sr = item['audio']['sampling_rate']
    if sr != 16000:
        audio_array = librosa.resample(audio_array, orig_sr=sr, target_sr=16000)
        sr = 16000
    
    audio_path = item['audio_path']
    audio_path = os.path.join(fleurs_configs['output_dir'], split, audio_path)
    os.makedirs(os.path.dirname(audio_path), exist_ok=True)

    item['audio_filepath'] = audio_path
    item['sampling_rate'] = sr
    item['duration'] = librosa.get_duration(y=audio_array, sr=sr)

    if os.path.isfile(audio_path):
        return item

    soundfile.write(audio_path, audio_array, sr, format='wav')

    return item

def process_item_hf(item, split, config):

    def parse_file_name():
        if config['filename_col'] in item:
            return item[config['filename_col']]
        # elif config['filename_col'] in item[config['audio_col']]:
        #     return item[config['audio_col']][config['filename_col']]
        return str(item['id']) + ".wav" # default to id if no filename column is found

    audio = item[config['audio_col']]
    # parse the audio from different types
    if isinstance(audio, dict) and 'array' in audio:
        audio_array = np.array(audio['array'], dtype=np.float32)
    elif isinstance(audio, str):
        audio_array, sr = librosa.load(audio)
    elif isinstance(audio, np.ndarray):
        audio_array = audio
        sr = config['sampling_rate']
    elif isinstance(audio, list):
        audio_array = np.array(audio, dtype=np.float32)
        sr = config['sampling_rate']
    else:
        audio_array = np.array(audio['array'], dtype=np.float32)
        sr = audio['sampling_rate']
    
    # resample if necessary
    if sr != TARGET_SR:
        audio_array = librosa.resample(audio_array, orig_sr=sr, target_sr=TARGET_SR)
        sr = TARGET_SR
    
    # save audio to disk
    audio_path = parse_file_name()
    audio_path = os.path.join(config['output_dir'], split, "clips", audio_path)
    os.makedirs(os.path.dirname(audio_path), exist_ok=True)

    item['audio_filepath'] = audio_path
    item['sampling_rate'] = TARGET_SR
    item['duration'] = librosa.get_duration(y=audio_array, sr=TARGET_SR)

    if os.path.isfile(audio_path):
        return item

    soundfile.write(audio_path, audio_array, TARGET_SR, format='wav')

    return item

def process_hf_dataset(config):
    for split in config['splits']:
        dataset = load_dataset(config['hf_dataset'], split=split, cache_dir=os.path.join(config['output_dir'], ".cache"))
        dataset = dataset.map(lambda x, idx: {"id": idx}, with_indices=True, num_proc=NUM_PROC)
        dataset = dataset.map(partial(process_item_hf, split=split, config=config), num_proc=NUM_PROC, desc=f"processing {split} dataset")
        
        if config['text_col'] != "text":
            dataset = dataset.rename_column(config['text_col'], "text")


        # dataset = dataset.rename_column(config['filename_col'], "audio_filepath")
        to_keep = ["audio_filepath", "text", "sampling_rate", "duration", "id"]
        dataset = dataset.remove_columns(set(dataset.column_names) - set(to_keep))

        split_metadata_path = os.path.join(config['output_dir'], split, f"{config['name']}_{split}_metadata.json")
        os.makedirs(os.path.dirname(split_metadata_path), exist_ok=True)
        with open(split_metadata_path, "w") as f:
            for item in tqdm(dataset):
                json.dump(item, f, ensure_ascii=False)
                f.write("\n")
        
        # clean up cache
        dataset.cleanup_cache_files()
        shutil.rmtree(os.path.join(config['output_dir'], ".cache"))
    return dataset

def main(args):


    # process hf datasets
    for dataset in args.datasets:
        eval(f"process_hf_dataset({dataset}_configs)")

    # # ArVoice
    # os.makedirs(arvoice_configs['output_dir'], exist_ok=True)
    # for split in arvoice_configs['splits']:
    #     total_arvoice = []
    #     prepocess_arvoice_fn = partial(process_arvoice, split=split)
    #     for csv_file in arvoice_configs['csv_files']:
    #         arvoice = load_dataset("csv", data_files=csv_file.format(split), cache_dir=arvoice_configs['output_dir'])["train"]
    #         arvoice = arvoice.map(lambda x:{"source": x["file_name"].split("/")[-3]}, num_proc=NUM_PROC)
    #         #TODO: filter out sample from khaleej source
    #         # arvoice = arvoice.filter(lambda x: x["source"] != "khaleej")
    #         arvoice = arvoice.map(prepocess_arvoice_fn, num_proc=NUM_PROC, desc=f"Processing {split} ArVoice dataset")
    #         total_arvoice.append(arvoice)
    #     total_arvoice = concatenate_datasets(total_arvoice)

    #     total_arvoice = total_arvoice.rename_column("transcription", "text")

    #     data_itr = iter(total_arvoice)
    #     split_metadata_path = os.path.join(arvoice_configs['output_dir'], split, f"arvoice_{split}_metadata.json")
    #     os.makedirs(os.path.dirname(split_metadata_path), exist_ok=True)
    #     with open(split_metadata_path, "w") as f:
    #         for item in tqdm(data_itr, desc=f"Dumping {split} ArVoice dataset"):
    #             if 'audio' in item:
    #                 del item['audio']
    #             json.dump(item, f, ensure_ascii=False)
    #             f.write("\n")
    # # clean up cache
    #     total_arvoice.cleanup_cache_files()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("datasets", type=str, nargs="+", choices=["clartts", "arvoice", "nadi", "tuneSwitch", "mixat", "fleurs"])
    args = parser.parse_args()
    main(args)



