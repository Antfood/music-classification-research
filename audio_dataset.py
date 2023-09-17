import torch
import torchaudio
import os
from torch.utils.data import Dataset
import pandas as pd

SAMPLE_RATE = 16000
X = 0
Y = 1

class AudioDataset(Dataset):
    def __init__(self, csv_path, transform=None, audio_len_seconds=10, sample_rate = SAMPLE_RATE, device="cpu"):
        assert os.path.isfile(csv_path), f"{csv_path} does not exist"

        self.labels = pd.read_csv(csv_path)
        self.sample_rate = SAMPLE_RATE
        self.transform = transform
        self.nb_samples = audio_len_seconds * self.sample_rate
        self.device = device

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        audio_path = self.labels.iloc[index, X]

        audio, sr = torchaudio.load(audio_path)  # type: ignore

        assert ( audio.shape[0] > 0 and audio.shape[1] > 0), f"Audio '{audio_path}' has no data. shape: {audio.shape}"

        if sr != SAMPLE_RATE:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            audio = resampler(audio)

        audio = self.to_mono(audio, audio_path)
        audio = self.pad(audio)
        audio = self.trim(audio)

        if self.transform:
            audio = self.transform(audio)

        label = self.labels.iloc[index, Y]

        return audio.to(self.device), label

    def trim(self, audio: torch.Tensor) -> torch.Tensor:
        if audio.shape[1] > self.nb_samples:
            audio = audio[:, : self.nb_samples]

        return audio


    def pad(self, audio: torch.Tensor) -> torch.Tensor:
        if audio.shape[1] < self.nb_samples:
            padding = (0, self.nb_samples - audio.shape[1])
            audio = torch.nn.functional.pad(audio, padding)

        return audio

    def to_mono(self, audio: torch.Tensor, audio_path: str) -> torch.Tensor:
        # muti-channel audio just pick first 2
        if audio.shape[0] > 2:
            channel = audio.shape[0]
            audio = audio[:2, :]
            print(
                f"WARNING: audio '{audio_path}' had {channel} channels. dropping {channel - 2} channel(s)."
            )

        if audio.shape[0] > 1:
            audio = torch.mean(audio, dim=0, keepdim=True)

        return audio
