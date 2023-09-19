import argparse
import pandas as pd
import os
import torch
import torchaudio
from transformers import Wav2Vec2Processor

def trim(audio, nb_samples) -> torch.Tensor:
    if audio.shape[1] > nb_samples:
        audio = audio[:, :nb_samples]

    return audio


def pad(audio, nb_samples) -> torch.Tensor:
    if audio.shape[1] < nb_samples:
        padding = (0, nb_samples - audio.shape[1])
        audio = torch.nn.functional.pad(audio, padding)

    return audio


def to_mono(audio, audio_path) -> torch.Tensor:
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

def parse_args():
    parser = argparse.ArgumentParser( description="""
                                     Preprocess audio files using Wav2Vec2 Wav2Vec2Processor
                                     By default, samples are trimmed to 10 seconds and resampled to 16kHz.
                                     """)
    parser.add_argument("csv_path", type=str, help="Path to input csv file")
    parser.add_argument("audio_dir", type=str, help="Path to audio directory")
    parser.add_argument( "out_dir", type=str, help="Output directory for preprocessed audio files")
    parser.add_argument( "--sr", type=int, default=16000, help="Sampling rate for audio files")
    parser.add_argument( "--max_dur", type=int, default=10, help="Maximum duration for audio files")

    return parser.parse_args()

## Main ## 

if __name__ == "__main__":
    args = parse_args()

    assert os.path.exists(args.csv_path), "Input csv file does not exist"
    assert os.path.exists(args.audio_dir), "Audio directory does not exist"

    os.makedirs(args.out_dir, exist_ok=True)

    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
    df = pd.read_csv(args.csv_path)

    assert "audio_path" in df.columns, "'audio_path' column not found in input csv file"
    assert "label" in df.columns, "'label' column not found in input csv file"

    print(f':: Preprocessing {len(df)} audio files')
    for index, row in df.iterrows():
        audio_path = os.path.join(args.audio_dir, row["audio_path"])  # type: ignore
        label = row["label"]

        print(f"Processing '{audio_path}'...")

        audio_data, sr = torchaudio.load(audio_path)  # type: ignore

        if sr != args.sr:
            resampler = torchaudio.transforms.Resample(sr, args.sr)
            audio = resampler(audio_data)

        audio_data = to_mono(audio_data, audio_path)
        audio_data = trim(audio_data, args.max_dur * args.sr)
        audio_data = pad(audio_data, args.max_dur * args.sr)

        input_values = processor(
            audio_data,
            return_tensors="pt",
            sampling_rate=16000,
            pad=True,
            truncate=True,
        ).input_values

        input_values = input_values.squeeze(dim=0)
        new_filename = os.path.basename(audio_path).replace(".wav", ".pt")
        processed_path = os.path.join(args.out_dir, new_filename)

        torch.save(input_values, processed_path)
        df.at[index, "audio_path"] = processed_path

    out_csv_path = args.csv_path.split(".")[0] + "-preprocess.csv"
    df.to_csv(out_csv_path)



