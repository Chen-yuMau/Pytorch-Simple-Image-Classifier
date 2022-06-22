import os
from os import walk
import sys
import cv2

if( len(sys.argv)==1):
	readtraindatapath = "MyData/raw/train/"
	writetraindatapath = "MyData/processed/train/"
	if not os.path.exists("MyData/processed/"):
		os.mkdir("MyData/processed") 
	if not os.path.exists("MyData/processed/train/"):
		os.mkdir("MyData/processed/train/")
	readtestdatapath = "MyData/raw/test/"
	writetestdatapath = "MyData/processed/test/"
	if not os.path.exists("MyData/processed/test/"):
		os.mkdir("MyData/processed/test/")
else:
	readdatapath = sys.argv[1]

file = open(readtraindatapath+"trainlabels.csv","w") 
labelnames = open(readtraindatapath+"trainlabelsmap.txt","w") 
i = 0
d = []
f = []
for (dirpath, dirnames, filenames) in walk(readtraindatapath):
	if i == 0:
		d = dirnames
		j = 0
		for n in d:
			labelnames.write(str(j)+": \""+n+"\",\n")
			j+=1
	else:
		f = filenames
		for n in f:
			file.write(n+","+str(i-1)+"\n")
			# print(readtraindatapath+d[i-1])
			img = cv2.imread(readtraindatapath+d[i-1]+"/"+n)
			newimg = cv2.resize(img,(250,250))
			if not os.path.exists(writetraindatapath+d[i-1]):
				os.mkdir(writetraindatapath+d[i-1])
			cv2.imwrite(writetraindatapath+d[i-1]+"/"+n,newimg)
	i+=1
labelnames.close()
file.close() 

file = open(readtestdatapath+"testlabels.csv","w") 
labelnames = open(readtestdatapath+"testlabelsmap.txt","w") 
i = 0
d = []
f = []
for (dirpath, dirnames, filenames) in walk(readtestdatapath):
	if i == 0:
		d = dirnames
		j = 0
		for n in d:
			labelnames.write(str(j)+": \""+n+"\",\n")
			j+=1
	else:
		f = filenames
		for n in f:
			file.write(n+","+str(i-1)+"\n")
			# print(readtestdatapath+d[i-1])
			img = cv2.imread(readtestdatapath+d[i-1]+"/"+n)
			newimg = cv2.resize(img,(250,250))
			cv2.imwrite(writetestdatapath+n,newimg)
	i+=1
labelnames.close()
file.close() 