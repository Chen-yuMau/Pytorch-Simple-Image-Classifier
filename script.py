from colormap import rgb2hex
import math
import time
cstart = 512
cend = 600
diff_mode = 'add' # 'add' or 'mult'
diff = 100
epochs = 100 #cant be less than 5
av = 1
starting_learning_rate = 1e-3
learning_rate_limit = 1e-3
datafile = 'output'
learning_curve_file = 'curveoutput'

add_timestats = [-5.444242377629945e-09, 1.0535968028132666e-05, 0.00030634068984983654, 2.0304991746169887]
def func(x, a, b, c, d):
    return a*(x**3) + b*(x**2) +c*x +d


est = 0
if diff_mode == 'add':
	est = ((func(cstart, *add_timestats)+func(cend,*add_timestats))*((cend-cstart)/diff)/2)*epochs*av
elif diff_mode == 'mult':	
	est = math.log((cend/cstart),diff)*av*epochs*(((cend+cstart)/200)+2)/2

print(f"Estimated time {time.strftime('%H:%M:%S', time.gmtime(est))}")
with open("Mass_Train.py","w") as f:
	f.write(f'##################################################################################################\n')
	f.write(f'import torch\n')
	f.write(f'import torch.nn as nn\n')
	f.write(f'import torch.nn.functional as F\n')
	f.write(f'import torch.optim as optim\n')
	f.write(f'import torch.optim.lr_scheduler as lr_scheduler\n')
	f.write(f'from torch.optim.lr_scheduler import _LRScheduler\n')
	f.write(f'import torch.utils.data as data\n')
	f.write(f'\n')
	f.write(f'import torchvision.transforms as transforms\n')
	f.write(f'import torchvision.datasets as datasets\n')
	f.write(f'import torchvision.models as models\n')
	f.write(f'\n')
	f.write(f'from sklearn import decomposition\n')
	f.write(f'from sklearn import manifold\n')
	f.write(f'from sklearn.metrics import confusion_matrix\n')
	f.write(f'from sklearn.metrics import ConfusionMatrixDisplay\n')
	f.write(f'import matplotlib.pyplot as plt\n')
	f.write(f'import numpy as np\n')
	f.write(f'\n')
	f.write(f'import copy\n')
	f.write(f'from collections import namedtuple\n')
	f.write(f'import os\n')
	f.write(f'import random\n')
	f.write(f'import shutil\n')
	f.write(f'import time\n')
	f.write(f'				\n')
	f.write(f'timestart = time.perf_counter()\n')
	f.write(f'\n')
	f.write(f'infile = \'long.pth\'\n')
	f.write(f'outfile = \'test.pth\'\n')
	f.write(f'new = True\n')
	f.write(f'epochs = {epochs}\n')
	f.write(f'starting_learning_rate = {starting_learning_rate}\n')
	f.write(f'learning_rate_limit = {learning_rate_limit}\n')
	f.write(f'\n')
	f.write(f'ROOT = \'MyData\'\n')
	f.write(f'data_dir = os.path.join(ROOT, \'processed\')\n')
	f.write(f'train_dir = os.path.join(data_dir, \'train\')\n')
	f.write(f'test_dir = os.path.join(data_dir, \'test\')\n')
	f.write(f'train_data = datasets.ImageFolder(root = train_dir, transform = transforms.ToTensor())\n')
	f.write(f'test_data = datasets.ImageFolder(root = test_dir, transform = transforms.ToTensor())\n')
	f.write(f'print(f\'Number of training examples: {{len(train_data)}}\')\n')
	f.write(f'print(f\'Number of testing examples: {{len(test_data)}}\')\n')
	f.write(f'BATCH_SIZE = 64\n')
	f.write(f'train_iterator = data.DataLoader(train_data, shuffle = True, batch_size = BATCH_SIZE)\n')
	f.write(f'test_iterator = data.DataLoader(test_data, batch_size = BATCH_SIZE)\n')
	f.write(f'device = "cuda" if torch.cuda.is_available() else "cpu"\n')
	f.write(f'print(f"Using {{device}} device")\n')
	f.write(f'with open("{datafile}.csv","w") as f:\n\n')
	f.write(f'	f.write(f\'n,accuracy,epoch time\\n\')\n\n')

	f.write(f'##################################################################################################\n')
	f.write(f'	with open("{learning_curve_file}.csv","w") as l:\n\n')
	c = cstart
	while c<=cend:
		f.write(f'		class NeuralNetwork{c}(nn.Module):\n')
		f.write(f'			def __init__(self):\n')
		f.write(f'				super(NeuralNetwork{c}, self).__init__()\n')
		f.write(f'				self.flatten = nn.Flatten()\n')
		f.write(f'				self.linear_relu_stack = nn.Sequential(\n')
		f.write(f'					nn.Linear(3*250*250, {c}),\n')
		f.write(f'					nn.ReLU(),\n')
		f.write(f'					nn.Linear({c}, {c}),\n')
		f.write(f'					nn.ReLU(),\n')
		f.write(f'					nn.Linear({c}, {c}),\n')
		f.write(f'					nn.ReLU(),\n')
		f.write(f'					nn.Linear({c}, 5),\n')
		f.write(f'				)\n')
		f.write(f'		\n')
		f.write(f'			def forward(self, x):\n')
		f.write(f'				x = self.flatten(x)\n')
		f.write(f'				logits = self.linear_relu_stack(x)\n')
		f.write(f'				return logits\n')
		f.write(f'\n\n')

		f.write(f'		l.write(f\'{c},\')\n\n')
		if diff_mode == 'add':
			c+=diff
		elif diff_mode == 'mult':
			c*=diff
	f.write(f'		l.write(f\'\\n\')\n')

	f.write(f'##################################################################################################\n')

	f.write(f'	def train_loop(dataloader, model, loss_fn, optimizer):\n')
	f.write(f'		size = len(dataloader.dataset)\n')
	f.write(f'		for batch, (X, y) in enumerate(dataloader):\n')
	f.write(f'			# Compute prediction and loss\n')
	f.write(f'			pred = model(X)\n')
	f.write(f'			loss = loss_fn(pred, y)\n')
	f.write(f'			# Backpropagation\n')
	f.write(f'			optimizer.zero_grad()\n')
	f.write(f'			loss.backward()\n')
	f.write(f'			optimizer.step()\n')
	f.write(f'	\n')
	f.write(f'			if batch % 100 == 0:\n')
	f.write(f'				loss, current = loss.item(), batch * len(X)\n')
	f.write(f'				print(f"loss: {{loss:>7f}}	[{{current:>5d}}/{{size:>5d}}]")\n')
	f.write(f'	\n')
	f.write(f'	def test_loop(dataloader, model, loss_fn):\n')
	f.write(f'		size = len(dataloader.dataset)\n')
	f.write(f'		num_batches = len(dataloader)\n')
	f.write(f'		test_loss, correct = 0, 0\n')
	f.write(f'	\n')
	f.write(f'		with torch.no_grad():\n')
	f.write(f'			for X, y in dataloader:\n')
	f.write(f'				pred = model(X)\n')
	f.write(f'				test_loss += loss_fn(pred, y).item()\n')
	f.write(f'				correct += (pred.argmax(1) == y).type(torch.float).sum().item()\n')
	f.write(f'	\n')
	f.write(f'		test_loss /= num_batches\n')
	f.write(f'		correct /= size\n')
	f.write(f'		print(f"Test Error: \\n Accuracy: {{(100*correct):>0.1f}}%, Avg loss: {{test_loss:>8f}} \\n")\n\n')
	f.write(f'		return(correct)\n')

	f.write(f'	cx = []\n')
	f.write(f'	cy = []\n')
	f.write(f'	ttt = []\n')

	f.write(f'##################################################################################################\n')
	c = cstart
	while c<=cend:
		f.write(f'	##################################################################################\n')
		f.write(f'	model = NeuralNetwork')
		f.write(str(c))##
		f.write(f'()\n')
		f.write(f'	avi = 1\n')
		f.write(f'	allacc = []\n')
		f.write(f'	ffaverage = []\n')
		f.write(f'	while avi<({av}+1):\n')
		f.write(f'		learning_rate = starting_learning_rate \n')
		f.write(f'		loss_fn = nn.CrossEntropyLoss()\n')
		f.write(f'		optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)\n')
		f.write(f'		\n')
		f.write(f'		acc = []\n')
		f.write(f'		epochtime = []\n')
		f.write(f'		for t in range(epochs):\n')
		f.write(f'			print(f"c = {c}")\n')
		f.write(f'			print(f"iteration {{avi}}")\n')
		f.write(f'			print(f"-------------------------------\\nEpoch {{t+1}}\\n-------------------------------")\n')
		f.write(f'			epochtstart = time.perf_counter()\n')
		f.write(f'			if len(acc)>2:\n')
		f.write(f'				if (acc[len(acc)-1]-acc[len(acc)-2])*(acc[len(acc)-2]-acc[len(acc)-3])<0:\n')
		f.write(f'					if learning_rate>learning_rate_limit:\n')
		f.write(f'						learning_rate*=(0.1**(1/3))\n')
		f.write(f'					optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)\n')
		f.write(f'			print("learning rate = "+str(learning_rate))\n')
		f.write(f'			train_loop(train_iterator, model, loss_fn, optimizer)\n')
		f.write(f'			acc.append(test_loop(test_iterator, model, loss_fn))\n')
		f.write(f'			epochtend = time.perf_counter()\n')
		f.write(f'			print(f"Epoch time spent {{(epochtend - epochtstart)/60:0.0f}} minutes {{(epochtend - epochtstart)%60:0.0f}} seconds\\n\\n")\n')
		f.write(f'			print(f"Total elapsed time {{(epochtend - timestart)/60/60:0.0f}} hours {{(epochtend - timestart)/60%60:0.0f}} minutes {{(epochtend - timestart)%60:0.0f}} seconds\\n\\n")\n')
		f.write(f'			epochtime.append(epochtend - epochtstart)\n')
		f.write(f'		\n')
		f.write(f'		\n')
		f.write(f'		i = 0\n')
		f.write(f'		sum = 0\n')
		f.write(f'		while i<5:\n')
		f.write(f'			i+=1\n')
		f.write(f'			sum += acc[len(acc)-i]\n')
		f.write(f'		faverage = sum/5\n')
		f.write(f'		print("final 5 average accuracy = "+str(faverage))\n')
		f.write(f'		ffaverage.append(faverage)\n')
		f.write(f'		timeend = time.perf_counter() # End Time\n')
		f.write(f'		# Print the Difference Minutes and Seconds\n')
		f.write(f'		print(f"Finished in {{(timeend - timestart)/60:0.0f}} minutes {{(timeend - timestart)%60:0.0f}} seconds")\n')
		f.write(f'		\n')
		f.write(f'		i = 0\n')
		f.write(f'		sum = 0\n')
		f.write(f'		while i<len(epochtime):\n')
		f.write(f'			sum += epochtime[i]\n')
		f.write(f'			i+=1\n')
		f.write(f'		epochtaverage = sum/len(epochtime)\n')
		f.write(f'		print(f"Average epoch time {{epochtaverage/60:0.0f}} minutes {{epochtaverage%60:0.0f}} seconds")\n')
		f.write(f'		\n')
		f.write(f'		allacc.append(acc)\n')
		f.write(f'		avi+=1\n')
		f.write(f'	ffaveragesum = 0\n')
		f.write(f'	for a in ffaverage:\n')
		f.write(f'		ffaveragesum+=a\n')
		f.write(f'	ffaveragesum/={av}\n')
		f.write(f'	cx.append({c})\n')
		f.write(f'	cy.append(ffaveragesum)\n')
		f.write(f'	ttt.append(epochtaverage)\n')


		f.write(f'	array = np.array(allacc)\n')
		f.write(f'	transposed_array = array.T\n')
		f.write(f'	Tacc = transposed_array.tolist()\n')
		f.write(f'	avacc = []\n')
		f.write(f'	for a in Tacc:\n')
		f.write(f'		asum = 0\n')
		f.write(f'		for b in a:\n')
		f.write(f'			asum+=b\n')
		f.write(f'		avacc.append(asum/{av})\n')
		f.write(f'	xcord = list(range(1,epochs+1))\n')

		f.write(f'	f.write(f\'{c},{{ffaveragesum}},{{epochtaverage}}\\n\')\n')
		color = c - cstart
		crange = (cend-cstart)/2
		if color > crange:
			rcolor = int((((2*crange)-color)*255)/crange)
			bcolor = 255
		else:
			bcolor = int(color*255/crange)
			rcolor = 255
		realcolor = rgb2hex(rcolor,0,bcolor)
		# print(f"({rcolor},0,{bcolor}) /  {realcolor} )")

		f.write(f'	plt.plot(xcord, avacc, \'{realcolor}\' )\n\n')
		if diff_mode == 'add':
			c+=diff
		elif diff_mode == 'mult':
			c*=diff
		f.write(f'	with open("{learning_curve_file}.csv","a") as l:\n\n')
		f.write(f'		for a in avacc:\n')
		f.write(f'			l.write(f\'{{a}},\')\n')
		f.write(f'		l.write(\'\\n\')\n')


		f.write(f'##################################################################################################\n')
	ts = time.strftime('%H:%M:%S', time.gmtime(est))
	f.write(f"	print(f\"Estimated time: {ts}\")\n")

	f.write(f'	plt.xlabel(\'Epoch\')\n')
	f.write(f'	plt.ylabel(\'Accuracy\')\n')
	f.write(f'	plt.show()\n')
	f.write(f'	plt.plot(cx, cy)\n')
	f.write(f'	plt.show()\n')
	f.write(f'	plt.plot(cx, ttt)\n')
	f.write(f'	plt.show()\n')
exec(open('Mass_Train.py').read())