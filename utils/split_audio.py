import os
import sys
import pandas as pd
from pydub import AudioSegment
from typing import List, Dict
import argparse

MAX_LEN = 10000  # milliseconds

def split_audio(filename, outdir, label, max_length) -> List[Dict[str, str]]:
    print("Splitting audio file: ", filename)

    splits = []

    audio = AudioSegment.from_wav(filename)

    if len(audio) <= max_length:
        output_file = os.path.join(outdir, os.path.basename(filename))
        audio.export(output_file, format="wav")
        print(f"{filename} is shorter than {max_length} milliseconds, no need to split")
        return []

    # Split the audio file
    num_chunks = len(audio) // max_length + 1

    for i in range(num_chunks):
        start_time = i * max_length
        end_time = (i + 1) * max_length if i != num_chunks - 1 else len(audio)
        chunk = audio[start_time:end_time]

        chunk_filename = os.path.join(
            outdir,
            f"{os.path.splitext(os.path.basename(filename))[0]}_part_{i + 1}.wav",
        )
        chunk.export(chunk_filename, format="wav")
        print(f":: Exported {chunk_filename}")
        split = {"audio_path": chunk_filename, "label": label}
        splits.append(split)

    return splits


def parse_args():
    parser = argparse.ArgumentParser(
        """Description: Split audio files into smaller chunks.
                        Exports a CVs with corresponding chucks and labels.
        """
    )
    parser.add_argument("csv_in", type=str, help="Path to the input CSV file")
    parser.add_argument("csv_out", type=str, help="Path to the output CSV file")
    parser.add_argument( "audio_out", type=str, help="Path where to output split audio file")
    parser.add_argument("label", type=str, help="label of the label column")
    parser.add_argument( "length", type=int, help="max length of the audio splits in SECONDS.")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if not os.path.exists(args.csv_in):
        print("CSV file does not exist")
        sys.exit(1)

    df = pd.read_csv(args.csv_in)

    if not os.path.exists(args.audio_out):
        os.mkdir(args.audio_out)

    if "audio_path" not in df.columns:
        print("Error: Input CSV file must have a column named 'audio_path'")
        exit(1)

    if args.label not in df.columns:
        print(f"Error: '{args.label}' column not found in input CSV file")
        exit(1)

    data = []

    for _, row in df.iterrows():
        try:
            splits = split_audio(
                row["audio_path"],
                args.audio_out,
                row[args.label],
                args.length * 1000
            )
            data.extend(splits)
        except:
            print(":::::::: Error splitting file: ", row["audio_path"], " skipping...")

    df = pd.DataFrame(data)
    df.to_csv(args.csv_out, index=False)
