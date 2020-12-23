import keras
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Softmax, Conv2D, Activation, Dropout, Flatten, AveragePooling2D
import matplotlib.pyplot as plt


(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

x_train /= 255
x_test /= 255
y_train = np_utils.to_categorical(y_train, 10)
y_test = np_utils.to_categorical(y_test, 10)

x_train = x_train.reshape(x_train.shape[0], 28,28,1)
x_test = x_test.reshape(x_test.shape[0], 28,28,1)

def create_model():
	model = Sequential()
	model.add(Conv2D(6, kernel_size=5, activation='tanh', input_shape=(28, 28, 1), padding='same'))
	model.add(AveragePooling2D((2, 2), strides=2))
	model.add(Conv2D(16, kernel_size=5, activation='tanh'))
	model.add(AveragePooling2D((2, 2), strides=2))
	model.add(Conv2D(120, kernel_size=5, activation='tanh'))
	model.add(Flatten())
	model.add(Dense(84, activation='tanh'))
	model.add(Dense(10, activation='softmax'))

	model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=['accuracy'])
	return model

model = create_model()
model.fit(x_train, y_train, epochs=10, batch_size=128, validation_data=(x_test, y_test))
model.save('model.h5')
