import torch
import torch.nn as nn
from torchaudio.transforms import MelSpectrogram

from mini_tortoise_tts.config import Config


class TorchMelSpectrogram(nn.Module):
    __slots__ = (
        "mel",
        "filter_length",
        "hop_length",
        "win_length",
        "n_mel_channels",
        "mel_fmin",
        "mel_fmax",
        "sampling_rate",
        "mel_stft",
    )

    _MEL_NORMS = None

    def __init__(
        self,
        filter_length: int = 1024,
        hop_length: int = 256,
        win_length: int = 1024,
        n_mel_channels: int = 80,
        mel_fmin: int = 0,
        mel_fmax: int = 8000,
        sampling_rate: int = 22050,
        normalize: bool = False,
    ):
        super().__init__()
        # These are the default tacotron values for the MEL spectrogram.
        self.filter_length = filter_length
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_mel_channels = n_mel_channels
        self.mel_fmin = mel_fmin
        self.mel_fmax = mel_fmax
        self.sampling_rate = sampling_rate
        self.mel_stft = MelSpectrogram(
            n_fft=self.filter_length,
            hop_length=self.hop_length,
            win_length=self.win_length,
            power=2,
            normalized=normalize,
            sample_rate=self.sampling_rate,
            f_min=self.mel_fmin,
            f_max=self.mel_fmax,
            n_mels=self.n_mel_channels,
            norm="slaney",
        )

    def forward(self, inp):
        # Automatically squeeze out the channels dimension if it is present (assuming mono-audio)
        if len(inp.shape) == 3:
            inp = inp.squeeze(1)

        if torch.backends.mps.is_available():
            inp = inp.to("cpu")

        self.mel_stft = self.mel_stft.to(inp.device)

        mel = self.mel_stft(inp)
        mel = torch.log(torch.clamp(mel, min=1e-5))

        mel_norms = self.get_default_mel_norms().to(mel.device)
        return mel / mel_norms.unsqueeze(0).unsqueeze(-1)

    def get_default_mel_norms(self):
        if self._MEL_NORMS is None:
            self._MEL_NORMS = torch.load(Config.default_mel_norm_file, weights_only=True)
        return self._MEL_NORMS
