##################################################################################################
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
				
timestart = time.perf_counter()

infile = 'long.pth'
outfile = 'test.pth'
new = True
epochs = 100
starting_learning_rate = 0.01
learning_rate_limit = 0.0001

ROOT = 'MyData'
data_dir = os.path.join(ROOT, 'processed')
train_dir = os.path.join(data_dir, 'train')
test_dir = os.path.join(data_dir, 'test')
train_data = datasets.ImageFolder(root = train_dir, transform = transforms.ToTensor())
test_data = datasets.ImageFolder(root = test_dir, transform = transforms.ToTensor())
print(f'Number of training examples: {len(train_data)}')
print(f'Number of testing examples: {len(test_data)}')
BATCH_SIZE = 64
train_iterator = data.DataLoader(train_data, shuffle = True, batch_size = BATCH_SIZE)
test_iterator = data.DataLoader(test_data, batch_size = BATCH_SIZE)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")
with open("output.csv","w") as f:

	f.write(f'n,accuracy,epoch time\n')

##################################################################################################
	with open("curveoutput.csv","w") as l:

		class NeuralNetwork512(nn.Module):
			def __init__(self):
				super(NeuralNetwork512, self).__init__()
				self.flatten = nn.Flatten()
				self.linear_relu_stack = nn.Sequential(
					nn.Linear(3*250*250, 512),
					nn.ReLU(),
					nn.Linear(512, 512),
					nn.ReLU(),
					nn.Linear(512, 512),
					nn.ReLU(),
					nn.Linear(512, 5),
				)
		
			def forward(self, x):
				x = self.flatten(x)
				logits = self.linear_relu_stack(x)
				return logits


		l.write(f'512,')

		l.write(f'\n')
##################################################################################################
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
				print(f"loss: {loss:>7f}	[{current:>5d}/{size:>5d}]")
	
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
	cx = []
	cy = []
	ttt = []
##################################################################################################
	##################################################################################
	model = NeuralNetwork512()
	avi = 1
	allacc = []
	ffaverage = []
	while avi<(1+1):
		learning_rate = starting_learning_rate 
		loss_fn = nn.CrossEntropyLoss()
		optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
		
		acc = []
		epochtime = []
		for t in range(epochs):
			print(f"c = 512")
			print(f"iteration {avi}")
			print(f"-------------------------------\nEpoch {t+1}\n-------------------------------")
			epochtstart = time.perf_counter()
			if len(acc)>2:
				if (acc[len(acc)-1]-acc[len(acc)-2])*(acc[len(acc)-2]-acc[len(acc)-3])<0:
					if learning_rate>learning_rate_limit:
						learning_rate*=(0.1**(1/3))
					optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
			print("learning rate = "+str(learning_rate))
			train_loop(train_iterator, model, loss_fn, optimizer)
			acc.append(test_loop(test_iterator, model, loss_fn))
			epochtend = time.perf_counter()
			print(f"Epoch time spent {(epochtend - epochtstart)/60:0.0f} minutes {(epochtend - epochtstart)%60:0.0f} seconds\n\n")
			print(f"Total elapsed time {(epochtend - timestart)/60/60:0.0f} hours {(epochtend - timestart)/60%60:0.0f} minutes {(epochtend - timestart)%60:0.0f} seconds\n\n")
			epochtime.append(epochtend - epochtstart)
		
		
		i = 0
		sum = 0
		while i<5:
			i+=1
			sum += acc[len(acc)-i]
		faverage = sum/5
		print("final 5 average accuracy = "+str(faverage))
		ffaverage.append(faverage)
		timeend = time.perf_counter() # End Time
		# Print the Difference Minutes and Seconds
		print(f"Finished in {(timeend - timestart)/60:0.0f} minutes {(timeend - timestart)%60:0.0f} seconds")
		
		i = 0
		sum = 0
		while i<len(epochtime):
			sum += epochtime[i]
			i+=1
		epochtaverage = sum/len(epochtime)
		print(f"Average epoch time {epochtaverage/60:0.0f} minutes {epochtaverage%60:0.0f} seconds")
		
		allacc.append(acc)
		avi+=1
	ffaveragesum = 0
	for a in ffaverage:
		ffaveragesum+=a
	ffaveragesum/=1
	cx.append(512)
	cy.append(ffaveragesum)
	ttt.append(epochtaverage)
	array = np.array(allacc)
	transposed_array = array.T
	Tacc = transposed_array.tolist()
	avacc = []
	for a in Tacc:
		asum = 0
		for b in a:
			asum+=b
		avacc.append(asum/1)
	xcord = list(range(1,epochs+1))
	f.write(f'512,{ffaveragesum},{epochtaverage}\n')
	plt.plot(xcord, avacc, '#FF0000' )

	with open("curveoutput.csv","a") as l:

		for a in avacc:
			l.write(f'{a},')
		l.write('\n')
##################################################################################################
	print(f"Estimated time: 00:06:38")
	plt.xlabel('Epoch')
	plt.ylabel('Accuracy')
	plt.show()
	plt.plot(cx, cy)
	plt.show()
	plt.plot(cx, ttt)
	plt.show()
