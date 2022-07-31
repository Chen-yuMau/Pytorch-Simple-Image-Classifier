import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.optim.lr_scheduler import _LRScheduler
import torch.utils.data as data

import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

from sklearn import decomposition
from sklearn import manifold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np

import copy
from collections import namedtuple
import os
import random
import shutil
import time

infile = 'long.pth'
outfile = 'test.pth'
new = True
epochs = 25
starting_learning_rate = 1e-3
learning_rate_limit = 1e-6
learning_rate = starting_learning_rate 

ROOT = 'MyData'
data_dir = os.path.join(ROOT, 'processed')
train_dir = os.path.join(data_dir, 'train')
test_dir = os.path.join(data_dir, 'test')

train_data = datasets.ImageFolder(root = train_dir, 
                                  transform = transforms.ToTensor())

test_data = datasets.ImageFolder(root = test_dir, 
                                  transform = transforms.ToTensor())


print(f'Number of training examples: {len(train_data)}')
print(f'Number of testing examples: {len(test_data)}')
# print(train_data.shape)
# print(test_data.shape)

BATCH_SIZE = 64

train_iterator = data.DataLoader(train_data, 
                                 shuffle = True, 
                                 batch_size = BATCH_SIZE)

test_iterator = data.DataLoader(test_data, 
                                batch_size = BATCH_SIZE)


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(3*250*250, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 14),
        )

    def forward(self, x):
        x = self.flatten(x)
        print(x.shape)
        logits = self.linear_relu_stack(x)
        return logits

if new == True:
	model = NeuralNetwork()#.to(device)
else:
	model = torch.load(infile)

# # X = torch.rand(3, 250, 250, device=device)
# X = disimg
# X = np.transpose(X, (2,1,0))
# X = X.to("cuda")
# logits = model(X)
# pred_probab = nn.Softmax(dim=1)(logits)
# y_pred = pred_probab.argmax(1)
# print(f"Predicted class: {y_pred}")

###################################################################################

batch_size = 64

def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return(correct)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

acc = []
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    if len(acc)>2:
    	if (acc[len(acc)-1]-acc[len(acc)-2])*(acc[len(acc)-2]-acc[len(acc)-3])<0:
    		learning_rate*=(0.1**(1/3))
    print("learning rate = "+str(learning_rate))
    train_loop(train_iterator, model, loss_fn, optimizer)
    acc.append(test_loop(test_iterator, model, loss_fn))

torch.save(model, outfile)

summ = 0
for alle in acc:
	summ+=alle
summ/= len(acc)
print("average = "+str(summ))

s = list(range(1,len(acc)-1))
k = []
for ss in s:
	k.append((acc[ss-1]+acc[ss]+acc[ss+1])/3)


xcord = list(range(1,epochs+1))
xxcord = list(range(2,epochs))
plt.plot(xcord, acc)
plt.plot(xxcord, k)
# plt.ylim(0, 0.5)  
plt.show()

# model = models.vgg16(pretrained=True)
ddd = []
for a in acc:
    ddd.append(int(100*a))
print(ddd)
print("Done!")


#[0.10714285714285714, 0.11428571428571428, 0.11190476190476191, 0.11904761904761904, 0.10952380952380952, 0.10476190476190476, 0.11190476190476191, 0.10952380952380952, 0.11190476190476191, 0.11666666666666667, 0.11428571428571428, 0.11666666666666667, 0.1261904761904762, 0.12380952380952381, 0.13095238095238096, 0.13333333333333333, 0.14523809523809525, 0.14523809523809525, 0.14523809523809525, 0.15]