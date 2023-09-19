import argparse
import torch
import os
import h5py

def parse_args():
    parser = argparse.ArgumentParser("Encodes prepossessed torch tensors (audio) into a h5py file")
    parser.add_argument("source_path", type=str, help="Path to torch tensors")
    parser.add_argument("out_path", type=str, help="Path to output h5py file")
    return parser.parse_args()

if __name__ == '__main__':

    args = parse_args()
    assert os.path.isdir(args.source_path), "Source path must be a directory"
    with h5py.File(args.out_path, 'w') as f:

        for filename in os.listdir(args.source_path):
            print(':: Processing file: ', filename)
            if filename.endswith(".pt"):
                tensor = torch.load(os.path.join(args.source_path, filename))
                f.create_dataset(filename.replace('.pt', ""), data=tensor.numpy(), compression="gzip")


