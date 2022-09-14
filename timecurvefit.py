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
    return a*(x**3) + b*(x**2) +c*x +d

with open(file, newline='') as f:
    reader = csv.reader(f)
    data = list(reader)
data.pop(0)

data1 = []
data2 = []
data3 = []
d3 = 0
for a in data:
	data1.append(int(a[0]))
	if len(a)>2:
		data2.append(float(a[2]))
		d3+= float(a[2])
		data3.append(d3)

x = np.array(data1)
y = np.array(data2)

popt,cov = curve_fit(func, x, y)

y_new_value = func(x, *popt)

plt.plot(x,y_new_value,color="red")
plt.plot(x,y)
plt.xlabel('N')
plt.ylabel('Seconds')
# print("Estimated value of a : "+ str(a))
# print("Estimated value of b : " + str(b))
print(*popt)
plt.show()





























