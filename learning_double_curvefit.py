import csv
import numpy as np
from numpy import mean
from numpy import std
from numpy import cov
from numpy.random import randn
from numpy.random import seed
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from matplotlib import pyplot
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

import matplotlib.pyplot as plt
file = '200_600_10-5learning_curve.csv'

def func(x, a, b, c, d,):
    # return a*(e**x) +b*x +c
    return a*(x**3) +b*(x**2) + c*(x) +d

def fun(x, a, e, b, c):
    # return a*(x**e) +b*x +c
    return a*(x**9) +b*(x**8) + c*(x**7) +e*(x**6)

with open(file, newline='') as f:
    reader = csv.reader(f)
    datay = list(reader)
datax = datay.pop(0)

#cast x into float
x = []
for a in datax:
	if a=='':
		continue
	x.append(float(a))
datax = x
#cast y into float
yy = []
for a in datay:
	y = []
	for b in a:
		if b == '':
			continue
		y.append(float(b))
	yy.append(y)
datay = yy

epochs = list(range(1,(len(datay[0])+1)))
#turn into np array
epochs = np.array(epochs)
datax = np.array(datax)
datay = np.array(datay)

new_coe = []
i = 0
for a in datay:
	popt,cov = curve_fit(func, epochs, a)
	new_coe.append(popt)
	y = func(epochs, *popt)
	plt.plot(epochs,y)
	plt.plot(epochs,a,color = 'red')

	plt.xlabel('Epochs')
	plt.ylabel('Accuracy')
	print(datax[i])
	print(popt)
	i+=1
plt.show()

new_coe = np.array(new_coe)
new_coe = new_coe.transpose()

coco = []
i = 0
for a in new_coe:
	plt.plot(datax,a)
	popt,cov = curve_fit(func, datax, a)
	coco.append(popt)
	y = func(datax, *popt)
	plt.plot(datax,y)
	plt.xlabel('N')
	if i==0:
		plt.ylabel('a')
		plt.ylim([-0.00015, 0.00015])
	if i==1:
		plt.ylabel('b')
		plt.ylim([-0.02, 0.02])
	if i==2:
		plt.ylabel('c')
		plt.ylim([-0.5, 0.5])
	if i==3:
		plt.ylabel('d')
		plt.ylim([-1, 1])
	plt.show()
	i+=1
for a in coco:
	print(a)

i = 0
for a in datay:

	y = func(epochs, func(datax[i],*coco[0]),func(datax[i],*coco[1]),func(datax[i],*coco[2]),func(datax[i],*coco[3]))
	fig, (ax1, ax2) = plt.subplots(1, 2)

	fig.suptitle('N = ' + str(datax[i]))
	ax1.plot(epochs,a,color = 'red')
	ax2.plot(epochs,y)

	ax1.set(xlabel='Original Data epoch')
	ax1.set(ylabel='Accuracy')
	ax2.set(xlabel='Predicted epoch')


	plt.show()
	i+=1






















