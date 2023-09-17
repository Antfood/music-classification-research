import os
import sys
import pandas as pd
from pydub import AudioSegment
from typing import List, Dict


def split_audio(
    filename, outdir, temp, max_length=10000
) -> List[Dict[str, str]]:  # max_length in milliseconds
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
        split = {"audio_path": chunk_filename, "temp": temp}
        splits.append(split)

    return splits


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Please provide the path to the csv file")
        exit()

    cvs_path = sys.argv[1]
    out_dir = "data/split_audio"

    df = pd.read_csv(cvs_path)

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    data = []

    for _, row in df.iterrows():
        try:
            splits = split_audio(row["audio_path"], out_dir, row["temp"])
            data.extend(splits)
        except:
            print(":::::::::::::: Error splitting file: ", row["audio_path"], " skipping...")

    df = pd.DataFrame(data)
    df.to_csv("data/split_audio.csv", index=False)
