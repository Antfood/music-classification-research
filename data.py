import os
import pprint as pprint
import pandas as pd
import boto3

"""
Downloads audio files from BP s3 bucket, saves them to data/audio and creates a csv file with the following columns:
    - recording: name of the recording
    - audio_path: path to the audio file
    - s3-key: name of the file in the s3 bucket
    - temp: temperature of the recording
"""

AUDIO_DIR = os.path.join(os.getcwd(), "data", "audio")
BUCKET_NAME = "betterproblems-production-originals"

# set number of files to download
NB_DOWNLOADS = 1000


def load_df(path):
    df = pd.read_csv(path)
    df.drop("Filename", axis="columns", inplace=True)
    df.dropna(how="all", inplace=True)
    df["Recording_S3_LocationLink"] = df["Recording_S3_LocationLink"].astype(str)
    df["s3-key"] = df["Recording_S3_LocationLink"].apply(lambda x: x.split("/")[-1])

    return df


def load_audio(df, s3, nb_files=10):
    count = 0
    out = []
    for _, row in df.iterrows():
        key = row["s3-key"]
        audio_path = os.path.join(AUDIO_DIR, key)
        print(f"Downloading: {key}")

        try:
            s3.download_file(BUCKET_NAME, key, audio_path)

            r = {
                "recording": row["Recordings"],
                "audio_path": audio_path,
                "s3-key": key,
                "temp": row["temp"],
            }

            out.append(r)
            count += 1

        except:
            print(f":::: Error downloading {key}")

        if count >= nb_files:
            break

    return pd.DataFrame(out)

if __name__ == "__main__":
    s3 = boto3.client("s3", region_name="us-east-1")
    df = load_df("data/temp-data.csv")
    df = load_audio(df, s3, nb_files=NB_DOWNLOADS)

    df.to_csv("data/recordings.csv", index=False)
