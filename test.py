import dataset
import helper
import transforms
import model
import meter
import time

from pathlib import Path
import torch
import torch.nn as nn
from torch.utils import data
from torch.utils.data.sampler import SubsetRandomSampler


#Hyper Paramater
LRATE = 0.01
BSIZE = 128
NUM_EPOCH = 100



path = 'dataset/iris.data'
feature_cols = ['sepal_length', 'sepal_width','petal_length','petal_witdh']
target_cols = ['class']
CLAZZ = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]

iris_dataset = dataset.IrisDataset(
    path, feature_cols,
    target_cols, CLAZZ,
    transforms_feature=transforms.NumpyToFloatTensor(),
    transforms_target=transforms.NumpyToLongTensor())

train_idx, valid_idx = helper.indice_splitter(iris_dataset, valid_size=0.2)

train_loader = data.DataLoader(iris_dataset, batch_size=BSIZE, sampler=SubsetRandomSampler(train_idx), num_workers=0)
valid_loader = data.DataLoader(iris_dataset, batch_size=BSIZE, sampler=SubsetRandomSampler(valid_idx), num_workers=0)

model = model.IrisNetwork(4,32,3)

#load weight from trained model
path = Path('./saved_model/checkpoint.pth')
checkpoint = torch.load(path)
model.load_state_dict(checkpoint['state_dict'])


viter = iter(valid_loader)
x_valid, y_valid = next(viter)
idx = 20

#get Network Output to Label
output = model(x_valid[idx])
softmax = torch.nn.functional.softmax(output, dim=0)
predicted = torch.argmax(softmax).item()
label_predict = iris_dataset.idx_to_class[predicted]

#Ground truth
ground_truth = y_valid[idx].item()
label_truth = iris_dataset.idx_to_class[ground_truth]

print(f'Network Predicted Result :\t{label_predict}\n'
      f'Ground Truth Result      :\t{label_truth}\n')
