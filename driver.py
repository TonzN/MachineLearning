import nn_v2 as nn
import random
import numpy as np
import tensorflow as tf
import plotly.express as px 
from matplotlib import pyplot


mnist = tf.keras.datasets.mnist

(trainX, trainY), (testX, testY) = mnist.load_data() 
trainX, testX = trainX / 255.0, testX / 255.0

test = nn.nNet()

test.FlattenLayer()
test.DenseLayer(196, 48, "relu") 
test.DenseLayer(48,10, "relu")

costs = []

dataset = []

batch = 12 
epochs = 12*120

for i in range(batch):
    dataset.append([0 for i in range(10)])
    dataset[i][trainY[i]] = 43

for x in range(epochs): 
    for i in range(batch):
        test.fit(nn.pool(trainX[i]), dataset[i], 0.000001)
        cost = nn.Cost(test.feedforward(nn.pool(trainX[i])), dataset[i])
        costs.append(cost)
      
    #print("EPOCH", x)

fig = px.scatter(x = costs, y = [i for i in range(epochs*batch)])
fig.show()

print("-----------------------------------------------------\n")
for i in range(12):
    print("PREDICT", np.argmax(test.feedforward(nn.pool(trainX[i]))))
    print("DATASET", dataset[i])
    print("IMG is number", trainY[i], "\n")
    
for i in range(1,12):
    pyplot.subplot(4,3, i+1)
    pyplot.imshow(trainX[i], cmap=pyplot.get_cmap('gray'))

pyplot.show()

