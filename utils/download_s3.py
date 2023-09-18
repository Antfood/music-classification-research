import os
import pprint as pprint
import pandas as pd
import boto3
import argparse

"""
Downloads audio files from BP s3 bucket, saves them to data/audio and creates a csv file with the following columns:
    - recording: name of the recording
    - audio_path: path to the audio file
    - s3-key: name of the file in the s3 bucket
    - temp: temperature of the recording
"""
BUCKET_NAME = "betterproblems-production-originals"

# set number of files to download
MAX_NB_DOWNLOADS = 10000

def parse_args():
    parser = argparse.ArgumentParser(description="Dowload audio files from s3 given a CSV file. Must contain a column named 'Recording_S3_LocationLink' with the s3 link to the audio file")
    parser.add_argument("csv_path", type=str, help="Path to the CSV file")
    parser.add_argument("audio_path", type=str, help="Path to the audio directory")

    return parser.parse_args()

def load_df(path):
    df = pd.read_csv(path)
    df.drop("Filename", axis="columns", inplace=True)
    df.dropna(how="all", inplace=True)
    df["Recording_S3_LocationLink"] = df["Recording_S3_LocationLink"].astype(str)
    df["s3-key"] = df["Recording_S3_LocationLink"].apply(lambda x: x.split("/")[-1])

    return df

def load_audio(df, s3, out_path, max_nb_files=10000):
    count = 0
    out = []
    for _, row in df.iterrows():
        key = row["s3-key"]
        audio_path = os.path.join(out_path, key)
        print(f"Downloading: {key}")

        try:
            s3.download_file(BUCKET_NAME, key, audio_path)

            r = row.to_dict()
            r['audio_path'] = audio_path
            r['s3-key'] = key

            out.append(r)
            count += 1

        except:
            print(f":::: Error downloading {key}")

        if count >= max_nb_files:
            break

    return pd.DataFrame(out)

if __name__ == "__main__":

    args = parse_args()

    if args.csv_path is None:
        raise ValueError("Must provide a path to the CSV file")

    if not os.path.exists(args.csv_path):
        raise ValueError(f"File not found: {args.csv_path}")

    if not os.path.exists(args.audio_path):
        print(f"Creating directory: '{args.audio_path}'")
        os.makedirs(args.audio_path)

    s3 = boto3.client("s3", region_name="us-east-1")
    df = load_df(args.csv_path)
    df = load_audio(df, s3, args.audio_path, max_nb_files=MAX_NB_DOWNLOADS)

    out_csv_path = args.csv_path.split(".")[0] + "-downloaded.csv"
    df.to_csv(out_csv_path, index=False)
