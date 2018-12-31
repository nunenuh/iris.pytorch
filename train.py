import dataset
import helper
import transforms
import model

import torch
import torch.nn as nn
from torch.utils import data
from torch.utils.data.sampler import SubsetRandomSampler


path = 'dataset/iris.data'
feature_cols = ['sepal_length', 'sepal_width','petal_length','petal_witdh']
target_cols = ['class']
clazz = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]

iris_dataset = dataset.IrisDataset(
    path, feature_cols,
    target_cols, clazz,
    transforms_feature=transforms.NumpyToFloatTensor(),
    transforms_target=transforms.NumpyToLongTensor())

train_idx, valid_idx = helper.indice_splitter(iris_dataset, valid_size=0.2)

train_loader = data.DataLoader(iris_dataset, batch_size=32, sampler=SubsetRandomSampler(train_idx), num_workers=0)
valid_loader = data.DataLoader(iris_dataset, batch_size=32, sampler=SubsetRandomSampler(valid_idx), num_workers=0)

net = model.IrisNetwork(4,32,3)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.01)

for epoch in range(100):
    for idx, (x_train, y_train) in enumerate(train_loader):
        out = net(x_train)
        loss = criterion(out, y_train)
        loss.backward()
        optimizer.step()
        print(loss)

    with torch.no_grad():
        for idx, (x_valid, y_valid) in enumerate(valid_loader):
            pass