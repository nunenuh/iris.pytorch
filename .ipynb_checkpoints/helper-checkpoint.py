import numpy as np
import torch
import shutil
from pathlib import Path

def indice_splitter(dataset, valid_size, shuflle=True):
    num_data = len(dataset)
    indices = list(range(num_data))
    split = int(np.floor(valid_size * num_data))
    if shuflle:
        np.random.seed(1)
        np.random.shuffle(indices)
    train_idx, valid_idx = indices[split:], indices[:split]
    return train_idx, valid_idx

def save_checkpoint(state, is_best, path='./saved_model/', filename='checkpoint.pth'):
    bpath = Path(path)
    fpath = bpath.joinpath(filename)
    torch.save(state, fpath)
    if is_best:
        shutil.copyfile(fpath, bpath.joinpath('model_best.pth'))