import argparse
import os
import shutil
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser(description='Reads a csv and copy only files listed in that csv to a directory of choice')

    parser.add_argument('csv_path', type=str, help="Path to csv file")
    parser.add_argument('source_dir', type=str, help="Path to source directory")
    parser.add_argument('dest_dir', type=str, help="Path to destination directory")

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    assert os.path.exists(args.csv_path), "CSV file does not exist" 
    assert os.path.exists(args.source_dir), "Source directory does not exist"
    assert os.path.exists(args.dest_dir), "Destination directory does not exist"

    df = pd.read_csv(args.csv_path)
    assert "audio_path" in df.columns, "CSV file does not contain audio_path column"

    for index, row in df.iterrows():
        try:
            output_path = os.path.join(args.dest_dir, row['audio_path']) # type: ignore
            input_path = os.path.join(args.source_dir, row['audio_path']) # type: ignore
            print(f":: Copying '{input_path}' -> '{output_path}'")
            shutil.copy(input_path, output_path)

        except Exception as e:
            print(f":::: ERROR: Could not copy {row['audio_path']} to {args.dest_dir}. Error: {e}")
            exit(1)

    print(":: Done")

