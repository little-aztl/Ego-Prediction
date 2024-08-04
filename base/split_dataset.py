import h5py, random
from tqdm import tqdm
import argparse

def split(h5py_path, test_ratio=0.3):
    data = h5py.File(h5py_path, "r+")
    seq_names = list(data.keys())
    random.shuffle(seq_names)
    test_seq_names = seq_names[:int(len(seq_names) * test_ratio)]
    train_seq_names = seq_names[int(len(seq_names) * test_ratio):]
    for test_name in tqdm(test_seq_names):
        data[test_name].attrs['train'] = False
    for train_name in tqdm(train_seq_names):
        data[train_name].attrs['train'] = True

    del data

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--h5py_path", type=str, default="./dataset/data.h5py")
    parser.add_argument("-r", "--test_ratio", type=float, default=0.3)
    args = parser.parse_args()

    split(args.h5py_path, args.test_ratio)