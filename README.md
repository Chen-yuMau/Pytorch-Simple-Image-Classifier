**ESE 599 - Research for Master's students**

Image classifier using MLP

Chen-yu Mau 113396995

**Introduction**

For this course, I was told to do research on MLP image classifiers on my own. I followed PyTorch documentation, used different datasets from the WikiArt dataset, and I also had to solve some problems of my own. In the end I searched for correlation as well as curvefitting to statistically analyze the data, and I came to some interesting results.

**Basic MLP image classifier**

In the beginning of this project, I had little to no experience with pytorch. I started with the pytorch tutorials. I decided that I wanted to use the most basic type NN, the multi -layer-perceptron. I learned about the following aspects:

Tensors

Datasets & DataLoader

Build the Neural Network

Optimizing Model Parameters

Save and Load the Model

The first thing I realized was that the images or the datasets that Mahan gave me are all different sized, so some pre-processing is needed. However, I didn't want my MLP to have to process the images every time I trained them, so I wrote this file to not only pre-process the images, but to also create the needed CSV files for the labels.

It is here I wrote **DatasetMaker.py**. The flow chart is as follows:

![Flowchart](https://github.com/Chen-yuMau/Pytorch-Simple-Image-Classifier/blob/main/Images/Flowchart1.png)

After that, I had some problems with the data loader that the tutorial advised me to use. The data loader defaults to treating the three-color matrices (RGB) as three separate grayscale images. This obviously isn't the effect that I wanted. After talking with Mahan, he advised me to use a different dataloader called Datafolder, and that has proven to be successful.

The following is the flow chart to the original **MLPTestModel.py** file that I wrote.

![Flowchart](https://github.com/Chen-yuMau/Pytorch-Simple-Image-Classifier/blob/main/Images/Flowchart2.png)

The original simple MLP definition is as follows:

self.linear\_relu\_stack = nn.Sequential(

nn.Linear(3\*250\*250, 512),

nn.ReLU(),

nn.Linear(512, 512),

nn.ReLU(),

nn.Linear(512, 512),

nn.ReLU(),

nn.Linear(512, 5),

)

Basically:

![Neural Network](https://github.com/Chen-yuMau/Pytorch-Simple-Image-Classifier/blob/main/Images/NN1.png)

At first I tried swapping the 512 with other numbers, however, a lot of the time I get the exact same accuracy or accuracies with the exact same intervals such as 33.4%, 41.7% and 50%. I realized that this was probably because I didn't have enough test cases in the dataset. Because it can only get an integer number of datapoints correct, if there aren't enough test cases the accuracy would look like it can only exist in certain increments. When I decided to download another dataset from Kaggle.com, everything finally started to look correct. (I used wikiart dataset. I chose the first 5 categories and 100 images for each category for training, and 30 for testing)

I also added a feature to lower the learning rate while training. I wrote a statement to determine whether if the accuracy was oscillating. If the accuracy was oscillating, this means the gradient decent step is too large and the learning rate is too high so I lower it a little bit. Obviously, until it hits the lower limit. When slope of accuracy between two consecutive intervals are different signs, multiply learning rate by 46.41%. (Lowering it 3 times is equal to multiply learning rate by 10%)

![Graph](https://github.com/Chen-yuMau/Pytorch-Simple-Image-Classifier/blob/main/Images/graph1.png)

Learning rate = 1e-3 Without lowering learning rate

![Graph](https://github.com/Chen-yuMau/Pytorch-Simple-Image-Classifier/blob/main/Images/graph2.png)

Starting Learning rate = 1e-2 Learning rate limit = 1e-4

Notice that the accuracy barely increases after 30 epochs.

**The Meta-Program**

Recall the defined MLP of last section.

I have since swapped the 512 nodes of each of the layers with a number N.

![Neural Network](https://github.com/Chen-yuMau/Pytorch-Simple-Image-Classifier/blob/main/Images/NN2.png)

I was experimenting with what number N could be that would make the accuracy higher. Since the accuracy barely increases after 30 epochs, I can train and test a small MLP like this in about 3 minutes. If I write a program to train and test 60 different MLPs it would only take 3 hours, and would be done in a reasonable amount of time. I realized that this way I can even plot out the **N to accuracy function** and I can gain insight to how this number N effects accuracy.I decided to write a "program that writes a program" or a "meta-program" called script.py to do this. This way I could easily bypass definition problems as well as object initialization problems.

![Flowchart](https://github.com/Chen-yuMau/Pytorch-Simple-Image-Classifier/blob/main/Images/FLowchart3.png)

I go into detail about this in the next section.

The "meta-program" I mentioned earlier is basically a "program that writes a program".

Because of the fact that there are many Neural Networks involved, and each one is defined differently, I decided to use this tactic to have a simpler view of things. As per the diagram below, Script.py outputs blocks of code according to parameters into Mass\_Train.py and then runs it.

![Chart](https://github.com/Chen-yuMau/Pytorch-Simple-Image-Classifier/blob/main/Images/chart.png)

If we can get a clean **N to accuracy function,** we can produce accurate predictions for the accuracy without even goin through the training prcedure.

**Statistical results**

After plotting the N to accuracy function, we can see that there is a huge amount of oscillation and noise. I even tried to train the same MLP three times and get an average and the result is basically the same.

![Graph](https://github.com/Chen-yuMau/Pytorch-Simple-Image-Classifier/blob/main/Images/graph_.png)

![Graph](https://github.com/Chen-yuMau/Pytorch-Simple-Image-Classifier/blob/main/Images/dataresults.png)

From the data I have gathered, we can make some conclusions.

With Pearsons correlation being 0.686, we can see that it has a generally positive and linear direction. With Spearman's correlation, we can also see that the two variables are somewhat monotonically related.

I then had the idea of doing a curve fit on the data that was produced. The module that I found was called Scipy. Within it, there is a function scipy.optimize.curve\_fit that uses non-linear least squares to fit a function, f, to the data. I decided to use the following exponential function to fit the data.

f(x) = a*(b^x)+cx+d

I chose this because there was an intuition that it was an exponential function. I decided to give a three more coefficients as additional modifiers. The results were as follows:

![Fitted Graph](https://github.com/Chen-yuMau/Pytorch-Simple-Image-Classifier/blob/main/Images/fittedcurvegraph.png)

With the previous coefficients :

a = -1.65583985e-01

b = 9.97982199e-01

c = -2.83427241e-05

d = 3.98636440e-01

If we plug it into our previous function:

f(x) = a*(b^x)+cx+d

We can use this to predict the theoretical maximum accuracy of the set of parameters.

On another note, even though the accuracy of the MLP is very random and scattered, the amount of time each epoch takes is very predictable and resembles a linear function.

![Fitted Graph](https://github.com/Chen-yuMau/Pytorch-Simple-Image-Classifier/blob/main/Images/timegraph.png)

**Additional experiments**

After I found the curve-fitting function, I had an additional idea:

I can curve fit every learning curve for many Ns and produce plots of N to Coefficients. If I do a curve fit a second time, I can produce 4 functions that can predict the learning curve coefficients of any given N. For this I decided to fit the curve using the basic polynomial:

N = 210

a = 1.59356302e-05

b = -1.06421645e-03

c = 2.30671954e-02

d = 1.70563523e-01

![Fitted Graph](https://github.com/Chen-yuMau/Pytorch-Simple-Image-Classifier/blob/main/Images/ga1.png)

N = 230

a = 2.76677074e-05

b = -1.63544796e-03

c = 3.04994865e-02

d = 1.80614974e-01

![Fitted Graph](https://github.com/Chen-yuMau/Pytorch-Simple-Image-Classifier/blob/main/Images/ga2.png)

N = 260

a = 1.40681952e-05

b = -9.10441274e-04

c = 1.93933048e-02

d = 1.68503556e-01

![Fitted Graph](https://github.com/Chen-yuMau/Pytorch-Simple-Image-Classifier/blob/main/Images/ga2.png)

**N to a Function** with curvefit

![Fitted Graph](https://github.com/Chen-yuMau/Pytorch-Simple-Image-Classifier/blob/main/Images/gb1.png)

**N to b Function** with curvefit

![Fitted Graph](https://github.com/Chen-yuMau/Pytorch-Simple-Image-Classifier/blob/main/Images/gb2.png)

**N to c Function** with curvefit

![Fitted Graph](https://github.com/Chen-yuMau/Pytorch-Simple-Image-Classifier/blob/main/Images/gb3.png)

**N to d Function** with curvefit

![Fitted Graph](https://github.com/Chen-yuMau/Pytorch-Simple-Image-Classifier/blob/main/Images/gb4.png)

Results:

![Fitted Graph](https://github.com/Chen-yuMau/Pytorch-Simple-Image-Classifier/blob/main/Images/gc1.png)

![Fitted Graph](https://github.com/Chen-yuMau/Pytorch-Simple-Image-Classifier/blob/main/Images/gc2.png)

I realize that almost every learning curve looks very similar and it results in very small differences in the coefficients and results in a lot of the same predicted curves.

**Conclusion**

I realize that many of these results are only for this specific case of MLP that I am using. number of the epochs, learning rate, input size, number of layers/neurons, optimizer..., there are much more parameters to train and experiment on. Even through I feel like the research I have done wasn't particularly conclusive, I still feel like I learned a lot about MLP just from experimenting and analyzing the data.
