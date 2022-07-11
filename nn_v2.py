from audioop import bias
from matplotlib.cbook import flatten
import numpy as np

def Cost(prediction, y): 
    cost = np.mean(np.square(y-prediction)) 
    return np.squeeze(cost)

def dCost(prediction, y): 
    return 2 * np.subtract(y, prediction)

#Activation functions
def softmax(x):
    tmp = np.exp(x)
    return tmp / np.sum(tmp)

def dSoftmax(x, lr):
    n = np.size(softmax(x))
    
def sigmoid(x):
    return 1/(1+np.exp(-x))

def dSigmoid(x):
    return sigmoid(x)*(1-sigmoid(x))

def relu(x):
    return np.maximum(0, x)

def dRelu(x):
    x[x<=0] = 0
    x[x>0] = 1
    return x

aFunctions = {
    "relu": relu,
    "sigmoid": sigmoid
} 
d_aFunctions = {
    "relu": dRelu,
    "sigmoid": dSigmoid
}

def Flatten(data):
    return data.flatten(order="A")
    
def pool(img):
    dim1 = int(img.shape[0]/2)
    dim2 = int(img.shape[1]/2)
    pool = np.array([np.zeros((dim1,dim2)) for i in range(1)]) 
    for z in range(len(pool)): #max pooling
        for i in range(dim1):
            for x in range(dim2): 
                pool[z][i][x] = np.amax(img[2*i:2+2*i, 2*x:2+2*x])
    return pool[0]

convolution = np.array([np.random.rand(3,3) for i in range(25)])
def convlayer(img):
    #runs one batch
    for i in range(len(convolution)):
        for z in range(25):
            for x in range(25):
                img[0+z:3+z, 0+x:3+x] = np.dot(img[0+z:3+z, 0+x:3+x],  convolution[i])
    return img

class nNet:
    def __init__(self):
        self.Weights = [] #list of weight matrices
        self.Biases = [] #lsit of bias vectors
        self.Layers = [] # list of activation functions
        self.modifiers = {} 

    def DenseLayer(self, inputDim, outputdim, activation): #activation has to be str
        #first layer inputdim = input layer 
        self.Weights.append(np.random.rand(inputDim, outputdim)) # rows = inpDim
        self.Biases.append(np.random.rand(outputdim)) 
        self.Layers.append(activation)
    
    def FlattenLayer(self):
        indx = len(self.Layers)-1
        if indx < 0:
            indx = 0

        self.modifiers[str(indx)] = Flatten

    def DrouputLayer(self, drouput):
        pass

    def activateLayer(self, input, layer):
        bias = self.Biases[layer]
        weights = self.Weights[layer]
        function = self.Layers[layer]
        sum = np.add(np.dot(input, weights), bias) # inp * weight + b

        return aFunctions[function](sum), sum 

    def feedforward(self, inp):
        self.log = {}
        self.log["A0"] = Flatten(inp)
        for i in range(len(self.Layers)): # i = layer
            if str(i) in self.modifiers:
                inp = self.modifiers[str(i)](inp) 
            prevActivation = inp
            inp, z = self.activateLayer(prevActivation, i)
            self.log["A"+str(i+1)] = inp
            self.log["z"+str(i)] = z
        return inp 

    def fit(self, inp, target, learningRate = 0.000001):
        predict = self.feedforward(inp)
        dcost = dCost(predict, target)
        self.deltalog = {}
        lastlayer = len(self.Layers)

        self.dBiases = []
        self.dWeights = []

        #Backprop
        for layerIndex in reversed(range(len(self.Layers))):
            dAF =  d_aFunctions[self.Layers[layerIndex]]  # derivative of current activation func
            z = self.log["z"+str(layerIndex)] 
            delta = np.multiply(dcost, dAF(z)) #delta * dA(z), dBias 
            delta_reshaped = delta.reshape(delta.shape[0], -1).T
            a_reshaped = self.log["A"+str(layerIndex)].reshape(self.log["A"+str(layerIndex)].shape[0], -1)
            self.dBiases.append(delta)
            self.dWeights.append(np.dot(a_reshaped, delta_reshaped))
           
            dcost = np.dot(delta, self.Weights[layerIndex].T)

        for layerIndex in range(len(self.Layers)):
            weights = self.Weights[len(self.Weights)-layerIndex-1]
            biases = self.Biases[layerIndex]
            self.Weights[len(self.Weights)-layerIndex-1] += learningRate*self.dWeights[layerIndex]
          #  print("_------------------------------",  weights - learningRate*self.dWeights[layerIndex], "---------------")
         #   print(self.Weights[len(self.Weights)-layerIndex-1])
        #   self.Biases[layerIndex] = np.subtract(biases, learningRate*self.dBiases[layerIndex])
    
    