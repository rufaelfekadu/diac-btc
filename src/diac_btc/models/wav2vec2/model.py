from diac_btc.models import DiacritizationModel
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC, AutoModelForCTC
import torch
import numpy as np
import librosa
import time



from diac_btc.text import (
    BASE_DIACRITICS,
    VALID_DIACRITICS_COMBINATIONS,
    preprocess_text,
    form_wildcard_pattern
)

class Wav2Vec2DiacritizationModel(DiacritizationModel):
    '''
    Wav2Vec2 diacritization model. implements
    - load_model
    - get_logits
    - diacritize ctc
    - diacritize wfst
    '''

    def __init__(self, model_name):
        self.model_name = model_name
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.sr = self.processor.feature_extractor.sampling_rate or 16000
        self.model = AutoModelForCTC.from_pretrained(model_name)
        self.device = torch.device("cpu")
        self.model.to(self.device)
        self.model.eval()

    @property
    def token2id(self):
        return dict(self.processor.tokenizer.get_vocab().items())

    @property
    def id2token(self):
        return {v: k for k, v in self.token2id.items()}

    @property
    def constrained_wildcard_ids(self):
        constrained_wildcard_ids = [self.token2id[ch] for ch in set(VALID_DIACRITICS_COMBINATIONS) if ch in self.processor.tokenizer.get_vocab()]
        constrained_wildcard_ids.append(self.token2id[self.processor.tokenizer.pad_token])
        ## TODO: Add no and unk diac tokens
        return constrained_wildcard_ids

    @property
    def unconstrained_wildcard_ids(self):
        return [k for k,v in self.processor.tokenizer.get_vocab().items()]

    @property
    def word_delimiter_token(self):
        return self.processor.tokenizer.word_delimiter_token

    @property
    def load_model(self):
        pass

    def get_logits(self, wav):
        '''
        Get logits from the model.
        Args:
            wav: wav file
        Returns:
            logits: logits from the model
        '''
        if isinstance(wav, np.ndarray):
            wav = torch.from_numpy(wav)
        elif isinstance(wav, torch.Tensor):
            wav = wav
        else:
            raise ValueError(f"Unsupported wav type: {type(wav)}")
       
        # if wav.dim() == 1:
        #     wav = wav.unsqueeze(0)

        inputs = self.processor(wav, sampling_rate=self.sr, return_tensors="pt", padding=True)

        with torch.no_grad():
            logits = self.model(inputs.input_values.to(self.device)).logits[0]
            log_probs = torch.log_softmax(logits, dim=-1)

        return log_probs

    def diacritize(self, text, audio_path, constrained=True, method="wfst"):
        # preprocess text
        text = preprocess_text(text)
        text = text.replace(" ", self.word_delimiter_token)
        # read audio and get audio array
        audio, sr = librosa.load(audio_path)
        if sr != self.sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sr)
            sr = self.sr

        # convert to mono
        if audio.ndim == 2:
            audio = audio[:, 0]

        audio = np.array(audio, dtype=np.float32)
        audio_duration = len(audio) / sr
        start_time = time.time()
        # get logits
        log_probs = self.get_logits(audio)
        pattern = form_wildcard_pattern(text)

        
        if method == "wfst":
            decoded = self.decode_wfst(log_probs, pattern, constrained=constrained)
        elif method == "ctc":
            decoded = self.decode_ctc(log_probs)
        else:
            raise ValueError(f"Invalid method: {method}")
        end_time = time.time()

        rtf = (end_time - start_time) / audio_duration
        # replace word delimiter with space
        if self.word_delimiter_token:
            decoded = decoded.replace(self.word_delimiter_token, " ")

        return decoded, rtf