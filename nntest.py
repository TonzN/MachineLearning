from math import ceil, floor
import nn_v2
import random
import numpy as np
import plotly.express as px 

dataset = [np.array([i]) for i in range(100)]

target = [x**2 for x in range(100)]

print("DATASET", dataset, "\n")
print("TARGET", target, "\n")

myNeuralNetwork = nn_v2.nNet()

myNeuralNetwork.DenseLayer(1, 5, "relu")
myNeuralNetwork.DenseLayer(5, 1, "relu")

def train(l):
    costs = []
    for i in range(l):
        for x in range(10):
            myNeuralNetwork.fit(dataset[x], target[x], 0.001)
            if len(costs) < 2000:
                costs.append(nn_v2.Cost(myNeuralNetwork.feedforward(dataset[x]), target[x]))
       # testn = random.randrange(0, 10)
        #print(nn_v2.Cost(myNeuralNetwork.feedforward(dataset[testn]), target[testn]))
    figl = l*10
    if figl > 2000:
        figl = 2000
    fig = px.scatter(x = costs, y = [i for i in range(figl)])
    fig.show()

run = True

while run:
    inp = input("pick a number to double: \n")
    if inp == "train":
        leng = input("length? ")
        train(int(leng))
    
    elif inp.isdigit():
        print("\nPREDICT: ", round(myNeuralNetwork.feedforward(float(inp))[0][0], 1))

