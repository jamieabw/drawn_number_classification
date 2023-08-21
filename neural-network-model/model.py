# import needed things
import tensorflow as tf
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt
import pickle

# declare constants, CHANGING THESE CAN MAKE THE MODEL LESS ACCURATE OR TAKE MORE TIME TO TRAIN
EPOCHS = 15 # how many times the model is trained on the training images
HLAYER_NEURONES = [256] # hidden layers neurones, if you add another layer, add another index of the amount of neurones
HLAYER_ACTIVATION = ["relu"] # the mathematical activation function used on each hidden layer

data = keras.datasets.mnist # dataset from keras to train the model on handwritten digits
(train_images, train_labels), (test_images, test_labels) = data.load_data() # grouping data between test and training
train_images = train_images / 255 # make each pixel colour value a number between 0 and 1 inclusive
test_images = test_images / 255
model = keras.Sequential( [
    keras.layers.Conv2D(32, (3,3), activation="relu", input_shape=(28, 28, 1)),
    keras.layers.MaxPool2D((2,2)),
    keras.layers.Conv2D(32, (3,3), activation="relu"), # convolutional layers
    keras.layers.Flatten(), 
    keras.layers.Dense(HLAYER_NEURONES[0], activation=HLAYER_ACTIVATION[0]),
    keras.layers.Dense(10, activation="sigmoid")
])
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.fit(train_images, train_labels, epochs=EPOCHS)
model.predict(test_images)
model.save("neural-network-model/model.h5")  # saves the models progress so it doesnt have to retrain in the actual program
