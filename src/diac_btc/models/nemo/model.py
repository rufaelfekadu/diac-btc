from diac_model import DiacritizationModel
from nemo.collections.asr.models import ASRModel
from nemo.collections.common.data.utils import move_data_to_device
from omegaconf import open_dict
import torch

class NemoDiacritizationModel(DiacritizationModel):
    '''
    Nemo diacritization model. implements
    - load_model
    - get_logits
    - diacritize ctc
    - diacritize wfst
    '''

    def __init__(self, model_path):
        self.model_path = model_path
        self.load_model()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_model(self):
        self.model = ASRModel.restore_from(self.model_path, map_location="cpu")

    def get_logits(self, wav, return_hypotheses=False):
        '''
        Get logits from the model.
        Args:
            wav: wav file
        Returns:
            logits: logits from the model
        '''
        wav = torch.from_numpy(wav).unsqueeze(0)
        length = torch.tensor([len(wav)])
        wav = move_data_to_device(wav, self.device)
        length = move_data_to_device(length, self.device)
        encoded, encoded_length  = self.model.forward(input_signal=wav, input_signal_length=length)
        logits = self.model.ctc_decoder(encoder_output=encoded)
        if return_hypotheses:
            hypotheses = self.model.ctc_decoding.ctc_decoder_predictions_tensor(
                logits, encoded_length, return_hypotheses=True,
            )
            return logits, hypotheses
        else:
            return logits

    def diacritize_ctc(self):
        pass

    def diacritize_wfst(self, text):
        '''
        Diacritize text using WFST.
        Args:
            text: text to diacritize
        Returns:
            diacritized text
        '''
    def diacritize(self):
        pass