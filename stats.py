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
file = '10_1200_10-5.csv'

def func(x, a, b, c, d):
    return a * (b**x) +c*x +d

with open(file, newline='') as f:
    reader = csv.reader(f)
    data = list(reader)
data.pop(0)

data1 = []
data2 = []
data3 = []
for a in data:
	data1.append(int(a[0]))
	data2.append(float(a[1]))
	if len(a)>2:
		data3.append(float(a[2]))

# plt.plot(x,y)
# plt.show()

# summarize
print('N: mean=%.3f stdv=%.3f' % (mean(data1), std(data1)))
print('Accuracy: mean=%.3f stdv=%.3f' % (mean(data2), std(data2)))
# plot
print()
covariance = cov(data1, data2)
print("covariance:")
for a in covariance:
	print(a)
print()
corr, _ = pearsonr(data1, data2)
print('Pearsons correlation: %.3f' % corr)

corr, _ = spearmanr(data1, data2)
print('Spearmans correlation: %.3f' % corr)

x = np.array(data1)
y = np.array(data2)

popt,cov = curve_fit(func, x, y)

# x_new_value = np.arange(min(x), 30, 5)
y_new_value = func(x, *popt)

plt.plot(x,y)
plt.plot(x,y_new_value,color="red")
plt.xlabel('N')
plt.ylabel('Accuracy')
# print("Estimated value of a : "+ str(a))
# print("Estimated value of b : " + str(b))
plt.show()
print(popt)
print()
print(cov)





























