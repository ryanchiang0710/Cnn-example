import matplotlib.pyplot as plt
import pandas as pd
from keras.datasets import mnist
import tensorflow as tf

(X_train_image, y_train_lable),(X_test_image, y_test_lable) = mnist.load_data()
print("Data image shape: ", X_train_image.shape)
print("First image sample:\n", X_train_image[0])
print("First sample data's label:", y_train_lable[0])

from keras.utils import to_categorical
X_train = X_train_image.reshape(60000,28,28,1)
X_test = X_test_image.reshape(10000,28,28,1)
y_train = to_categorical(y_train_lable, 10)
y_test = to_categorical(y_test_lable, 10)
print("Train shape: ", X_train.shape)
print("First data lable: ",y_train[0])

from keras import models
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
model = models.Sequential()
model.add(Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))
model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])

model.fit(X_train, y_train,validation_split=0.3,epochs=5,batch_size=128)

score = model.evaluate(X_test, y_test)
print("Model score: ", score[1])