import numpy as np

def indice_splitter(dataset, valid_size, shuflle=True):
    num_data = len(dataset)
    indices = list(range(num_data))
    split = int(np.floor(valid_size * num_data))
    if shuflle:
        np.random.seed(1)
        np.random.shuffle(indices)
    train_idx, valid_idx = indices[split:], indices[:split]
    return train_idx, valid_idx