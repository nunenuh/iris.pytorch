import torch
import numpy as np

class NumpyToFloatTensor(object):
    def __call__(self, param):
        return torch.from_numpy(param.astype(np.float32)).float()

class NumpyToLongTensor(object):
    def __call__(self, param):
        return torch.from_numpy(param.astype(np.long)).long()
