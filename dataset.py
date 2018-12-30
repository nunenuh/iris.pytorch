from torch.utils import data
from pathlib import Path
import pandas as pd
import numpy as np

class IrisDataset(data.Dataset):
    def __init__(
            self, path: str, feature_cols: list,
            target_cols: list, clazz: list,
            transforms_feature=None, transforms_target=None):
        self.path = Path(path)
        self.dframe = pd.read_csv(self.path)
        self.feature_cols = feature_cols
        self.target_cols = target_cols
        self.clazz = clazz
        self.transforms_feature = transforms_feature
        self.transforms_target = transforms_target

        self.__normalize_target()
        self.class_to_idx = self.__class_to_label()
        self.idx_to_class = self.__idx_to_class()

    def __len__(self):
        return len(self.dframe)

    def __class_to_label(self):
        mapz = [(val, idx) for idx, val in enumerate(self.clazz)]
        return dict(mapz)

    def __idx_to_class(self):
        mapz = [(idx, val) for idx, val in enumerate(self.clazz)]
        return dict(mapz)

    def __normalize_target(self):
        cat_type = CategoricalDtype(categories=self.clazz, ordered=True)
        self.dframe[self.target_cols[0]] = self.dframe[self.target_cols[0]].astype(cat_type).cat.codes

    def __getitem__(self, idx):
        feature = self.dframe[self.feature_cols].iloc[idx].values
        target = self.dframe[self.target_cols].iloc[idx].values
        target = np.squeeze(target)

        if self.transforms_feature:
            feature = self.transforms_feature(feature)
        if self.transforms_target:
            target = self.transforms_target(target)

        return feature, target

