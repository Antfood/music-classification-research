import torchaudio
import os
import pandas as pd
import argparse

def parse_args():
    parser = argparse.ArgumentParser("Cleans up audio files that are corrupted or empty.")
    parser.add_argument("csv_path", type=str, help="path to csv file. Must contain a 'audio_path' column.")
    return parser.parse_args()

    
if __name__ == "__main__":

    args = parse_args()
    assert os.path.exists(args.csv_path), 'csv_path does not exist'

    df = pd.read_csv("data/split_audio.csv")
    assert "audio_path" in df.columns, "CSV file must contain a column named 'audio_path'"

    drop_indices = []

    for index, row in df.iterrows():
        audio_path = row["audio_path"]
        audio, sr = torchaudio.load(audio_path)  # type: ignore

        if audio.shape[1] == 0:
            print(
                f"Audio '{audio_path}' has no data. shape: {audio.shape}. Dropping row."
            )
            drop_indices.append(index)

    # Drop rows with empty audio files
    df.drop(drop_indices, inplace=True)

    # Save the updated DataFrame to a new CSV file
    out_csv_path = args.csv_path.split(".")[0] + "-cleaned.csv"
    df.to_csv(out_csv_path, index=False)
