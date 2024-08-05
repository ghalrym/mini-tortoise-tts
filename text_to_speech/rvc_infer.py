import os,sys

from rvc.memory_config import Config
from rvc.util import load_hubert_model

now_dir = os.getcwd()
sys.path.append(now_dir)
import sys
import torch
import numpy as np
import yaml
import logging

from vc_infer_pipeline import VC
from lib.infer_pack.models import SynthesizerTrnMs256NSFsid, SynthesizerTrnMs256NSFsid_nono, SynthesizerTrnMs768NSFsid, SynthesizerTrnMs768NSFsid_nono
from rvc.audio import load_audio

from scipy.io import wavfile


logging.getLogger('fairseq').setLevel(logging.ERROR)
logging.getLogger('rvc').setLevel(logging.ERROR)



def load_config():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    yaml_file = os.path.join(current_dir, "rvc.yaml")

    with open(yaml_file, "r") as file:
        rvc_conf = yaml.safe_load(file)

    return rvc_conf


class CudaNotAvailable(Exception):
    def __init__(self):
        Exception.__init__(self, "Cuda or MPS not detected")


class VoiceConversion:
    __slots__ = (
        "config",
        "hubert_model",
        "vc",
        "tgt_sr",
        "n_spk",
        "net_g",
    )

    def __init__(
        self,
        model_path: str,
        is_half: bool = False,
    ):
        if torch.cuda.is_available():
            device = "cuda:0"
        elif torch.backends.mps.is_available():
            device = "mps:0"
        else:
            raise CudaNotAvailable()

        self.config = Config(device, is_half)
        self.hubert_model = load_hubert_model(self.config)

        self.vc, self.tgt_sr, self.n_spk, self.net_g = self._setup_vc(model_path)

    def _setup_net_g(self, version: str, cpt: dict, if_f0):
        if version == "v1":
            if if_f0 == 1:
                net_g = SynthesizerTrnMs256NSFsid(*cpt["config"], is_half=self.config.is_half)
            else:
                net_g = SynthesizerTrnMs256NSFsid_nono(*cpt["config"])
        elif version == "v2":
            if if_f0 == 1:
                net_g = SynthesizerTrnMs768NSFsid(*cpt["config"], is_half=self.config.is_half)
            else:
                net_g = SynthesizerTrnMs768NSFsid_nono(*cpt["config"])

        del net_g.enc_q

        net_g.eval().to(self.config.device)
        net_g = net_g.half() if self.is_half else net_g.float()
        return net_g

    def _setup_vc(self, model_path: str):
        # Loading PTH
        cpt = torch.load(model_path, map_location="cpu")
        cpt["config"][-3] = cpt["weight"]["emb_g.weight"].shape[0]  # n_spk
        net_g = self._setup_net_g(cpt.get("version", "v1"), cpt, cpt.get("f0", 1))

        return VC(tgt_sr, self.config), cpt["config"][-1], cpt["config"][-3], net_g

    def _vc_single(
        self,
        sid,
        input_audio_path,
        f0_up_key,
        f0_file,
        f0_method,
        file_index,
        file_index2,
        # file_big_npy,
        index_rate,
        filter_radius,
        resample_sr,
        rms_mix_rate,
        protect,
    ):  # spk_item, input_audio0, vc_transform0,f0_file,f0method0
        global net_g, vc, hubert_model, version
        f0_file = None
        if input_audio_path is None:
            return "You need to upload an audio", None
        f0_up_key = int(f0_up_key)
        audio = load_audio(input_audio_path, 16000)
        audio_max = np.abs(audio).max() / 0.95
        if audio_max > 1:
            audio /= audio_max
        times = [0, 0, 0]

        if_f0 = self.cpt.get("f0", 1)
        file_index = (
            (
                file_index.strip(" ")
                .strip('"')
                .strip("\n")
                .strip('"')
                .strip(" ")
                .replace("trained", "added")
            )
            if file_index != ""
            else file_index2
        )

        audio_opt = vc.pipeline(
            hubert_model,
            self.net_g,
            sid,
            audio,
            input_audio_path,
            times,
            f0_up_key,
            f0_method,
            file_index,
            # file_big_npy,
            index_rate,
            if_f0,
            filter_radius,
            self.tgt_sr,
            resample_sr,
            rms_mix_rate,
            version,
            protect,
            f0_file=f0_file,
        )
        return audio_opt

    def rvc_convert(
        self,
        model_path: str,
        f0_up_key: int = 0,
        input_path: str | None = None,
        output_dir_path: str | None = None,
        f0method: str = "rmvpe",
        file_index: str = "",
        file_index2: str = "",
        index_rate: int = 1,
        filter_radius: int = 3,
        resample_sr: int = 0,
        rms_mix_rate: float = 0.5,
        protect: float = 0.33,
    ):

        if output_dir_path is None:
            output_dir_path = "output"
            output_file_name = "out.wav"
            output_dir = os.getcwd()
            output_file_path = os.path.join(output_dir, output_dir_path, output_file_name)
        else:
            output_file_path = output_dir_path
            pass

        wav_opt = self._vc_single(
            0,
            input_path,
            f0_up_key,
            None,
            f0method,
            file_index,
            file_index2,
            index_rate,
            filter_radius,
            resample_sr,
            rms_mix_rate,
            protect
        )
        wavfile.write(output_file_path, tgt_sr, wav_opt)
        print(f"\nFile finished writing to: {output_file_path}")

        return output_file_path