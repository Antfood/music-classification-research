import torch
import torchaudio
import os
from torch.utils.data import Dataset
import pandas as pd
import logging

logging.basicConfig(
    level=logging.INFO, format="%[%(levelname)s]: %(message)s", datefmt="%H:%M:%S"
)

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000

class AudioDataset(Dataset):
    def __init__(
        self,
        csv_path,
        audio_path,
        y_col_name="label",
        x_col_name="audio_path",
        transform=None,
        audio_len_seconds=10,
        sample_rate=SAMPLE_RATE,
        device="cpu",
        cleanup=False,
    ):
        assert os.path.isfile(csv_path), f"{csv_path} does not exist"

        self.audio_path = audio_path
        self.y_col_name = y_col_name
        self.x_col_name = x_col_name
        self.labels = pd.read_csv(csv_path)
        self.sample_rate = sample_rate
        self.transform = transform
        self.nb_samples = audio_len_seconds * self.sample_rate
        self.device = device

        if cleanup:
            self.clean_up()  # drop rows that have no valid audio data

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        audio_filename = self.labels.iloc[index, self.x_col_name]
        audio_path = os.path.join(self.audio_path, audio_filename)

        audio, sr = torchaudio.load(audio_path)  # type: ignore

        assert (
            audio.shape[0] > 0 and audio.shape[1] > 0
        ), f"Audio '{audio_path}' has no data. shape: {audio.shape}"

        if sr != SAMPLE_RATE:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            audio = resampler(audio)

        audio = self._to_mono(audio, audio_path)
        audio = self._pad(audio)
        audio = self._trim(audio)

        if self.transform:
            audio = self.transform(audio)

        label = self.labels.iloc[index, self.y_col_name]

        return audio.to(self.device), label

    def _trim(self, audio: torch.Tensor) -> torch.Tensor:
        if audio.shape[1] > self.nb_samples:
            audio = audio[:, : self.nb_samples]

        return audio

    def _pad(self, audio: torch.Tensor) -> torch.Tensor:
        if audio.shape[1] < self.nb_samples:
            padding = (0, self.nb_samples - audio.shape[1])
            audio = torch.nn.functional.pad(audio, padding)

        return audio

    def _to_mono(self, audio: torch.Tensor, audio_path: str) -> torch.Tensor:
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

    # this method will drop rows that have no valid audio data
    def clean_up(self):
        drop_indices = []

        for index, row in self.labels.iterrows():
            audio_path = row[self.x_col_name]

            audio, _ = torchaudio.load(audio_path)  # type: ignore
            if audio.shape[1] == 0:
                msg = f"Audio '{audio_path}' has no data. shape: {audio.shape}. Dropping row."
                logger.warning(msg)
                drop_indices.append(index)

        self.labels.drop(drop_indices, inplace=True)
