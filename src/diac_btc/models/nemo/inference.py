import nemo.collections.asr as nemo_asr
from nemo.collections.asr.models import(
    ASRModel
)
from nemo.collections.common.data.utils import move_data_to_device
from omegaconf import open_dict
import torch
from pyarabic import araby
from jiwer import wer

def clean_text(text, remove_diacritics=False):
    if remove_diacritics:
        text = araby.strip_diacritics(text)
    # replace | and - with space
    text = text.replace('|', ' ').replace('-', ' ')
    # collapse extra spaces
    text = text.replace('  ', ' ')
    return text

def calculate_wer(hyp, ref):
    if isinstance(hyp, list):
        hyp = [clean_text(h) for h in hyp]
        ref = [clean_text(r) for r in ref]
    else:
        hyp = clean_text(hyp)
        ref = clean_text(ref)
    return wer(ref, hyp)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

asr_model = ASRModel.restore_from("/home/rufael/Projects/diac-btc/outputs/ckpts/stt_ar_fastconformer_hybrid_large_pcd_v1.0.nemo", map_location="cpu")
asr_model.to(device)

test_audio = "/home/rufael/Projects/diac-btc/notebooks/samples/female_ab_00000.wav"
test_reference = open("/home/rufael/Projects/diac-btc/notebooks/samples/female_ab_00000.txt", "r").read()

# Configure for greedy CTC
with open_dict(asr_model.cfg.decoding):
    asr_model.cfg.decoding.strategy = "greedy"
    asr_model.cfg.decoding.compute_timestamps = False # Optional
asr_model.change_decoding_strategy(decoder_type="ctc")

# test model
output = asr_model.transcribe([test_audio])

print(output[0].text)
print(test_reference)
print(f"WER: {calculate_wer(output[0].text, test_reference)}")