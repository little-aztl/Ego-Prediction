import h5py, random

def split(h5py_path, test_ratio=0.3):
    data = h5py.File(h5py_path, "r+")
    seq_names = list(data.keys())
    random.shuffle(seq_names)
    test_seq_names = seq_names[:int(len(seq_names) * test_ratio)]
    train_seq_names = seq_names[int(len(seq_names) * test_ratio):]
    for test_name in test_seq_names:
        data[test_name].attrs['train'] = False
    for train_name in train_seq_names:
        data[train_name].attrs['train'] = True

    del data