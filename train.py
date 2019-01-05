import dataset
import helper
import transforms
import model
import meter
import time

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
target_cols = ['species']
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
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=LRATE)

best_loss = 1.5
history = {'epoch':[], 'train_loss':[],'valid_loss':[],}
for epoch in range(NUM_EPOCH):
    batch_time = meter.AverageMeter()
    data_time = meter.AverageMeter()
    losses = meter.AverageMeter()

    end_time = time.time()
    for idx, (x_train, y_train) in enumerate(train_loader):
        data_time.update(time.time() - end_time)

        out = model(x_train)
        loss = criterion(out, y_train)
        loss.backward()
        optimizer.step()

        losses.update(loss.item(), x_train.size(0))
        batch_time.update(time.time() - end_time)
        end_time = time.time()

        print(f'Train Epoch [{epoch+1}/{NUM_EPOCH}] [{idx}/{len(train_loader)}]\t'
              f' Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
              f' Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
              f' Loss {losses.val:.4f} ({losses.avg:.4f}) ')

    history['epoch'].append(epoch)
    history['train_loss'].append(losses.avg)


    with torch.no_grad():
        end_time = time.time()
        for idx, (x_valid, y_valid) in enumerate(valid_loader):
            data_time.update(time.time() - end_time)

            out = model(x_valid)
            loss = criterion(out, y_valid)

            losses.update(loss.item(), x_valid.size(0))
            batch_time.update(time.time() - end_time)
            end_time = time.time()

            print(f'Valid Epoch [{epoch + 1}/{NUM_EPOCH}] [{idx}/{len(valid_loader)}]\t'
                  f' Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  f' Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  f' Loss {losses.val:.4f} ({losses.avg:.4f}) ')

    history['valid_loss'].append(losses.avg)

    is_best = losses.avg < best_loss
    best_loss = min(losses.avg, best_loss)
    helper.save_checkpoint({
        'epoch': epoch + 1,
        'batch_size': BSIZE,
        'learning_rate': LRATE,
        'total_clazz': len(CLAZZ),
        'class_to_idx': iris_dataset.class_to_idx,
        'labels': CLAZZ,
        'history': history,
        'arch': 'IrisNet',
        'state_dict': model.state_dict(),
        'best_loss': best_loss,
        'optimiz1er': optimizer.state_dict(),
    }, is_best)

