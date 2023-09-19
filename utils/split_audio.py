import os
import sys
import pandas as pd
from pydub import AudioSegment
from typing import List, Dict
import argparse

def split_audio(filename, indir, outdir, label, max_length) -> List[Dict[str, str]]:
    print("Splitting audio file: ", filename)

    splits = []

    in_path = os.path.join(indir, filename)
    audio = AudioSegment.from_wav(in_path)

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

        Usage:
            python split_audio.py <csv_in> <csv_out> <audio_in> <audio_out> <label> <length>
        """
    )
    parser.add_argument("csv_in", type=str, help="Path to the input CSV file")
    parser.add_argument("csv_out", type=str, help="Path to the output CSV file")
    parser.add_argument("audio_in", type=str, help="Path to directory input audio file")
    parser.add_argument( "audio_out", type=str, help="Path where to output split audio file")
    parser.add_argument("label", type=str, help="label of the label column")
    parser.add_argument( "length", type=int, help="max length of the audio splits in SECONDS.")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    assert os.path.exists(args.audio_in), "Error: Input audio directory does not exist"
    assert os.path.exists(args.csv_in), "Error: Input CSV file does not exist"

    df = pd.read_csv(args.csv_in)

    if not os.path.exists(args.audio_out):
        os.mkdir(args.audio_out)

    assert "audio_path" not in df.columns, "Error: Input CSV file must have a column named 'audio_path'"
    assert "label" not in df.columns, "Error: Input CSV file must have a column named 'label'"

    data = []

    for _, row in df.iterrows():
        try:
            splits = split_audio(
                row["audio_path"],
                args.audio_in,
                args.audio_out,
                row[args.label],
                args.length * 1000
            )
            data.extend(splits)
        except:
            print(":::::::: Error splitting file: ", row["audio_path"], " skipping...")

    df = pd.DataFrame(data)
    df.to_csv(args.csv_out, index=False)
