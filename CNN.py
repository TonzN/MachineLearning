from cmath import sqrt
import tensorflow as tf
from matplotlib import pyplot

plotting
#for i in range(1,10):  
 #   pyplot.subplot(3,3, i)
  #  pyplot.imshow(train_X[i], cmap=pyplot.get_cmap('gray'))
#pyplot.show()


mnist = tf.keras.datasets.mnist

(trainX, trainY), (testX, testY) = mnist.load_data() 

testX = testX.reshape((testX.shape[0], 28, 28, 1))
trainX = trainX.reshape((trainX.shape[0], 28, 28, 1))
print(trainY[1])
trainY = tf.keras.utils.to_categorical(trainY)
print(trainY[1])