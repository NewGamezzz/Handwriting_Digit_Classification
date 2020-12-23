from keras.models import load_model
from PIL import Image
import os
import matplotlib.pyplot as plt
import numpy as np
import cv2

def predict(path):
	model = load_model('model.h5')

	img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
	img = Image.fromarray(img)
	img = img.resize((28, 28))
	img = np.array(img).reshape((1, 28, 28, 1))
	# print(img.shape)
	# fig, ax = plt.subplots()
	# ax.imshow(img[0, :, :, 0], cmap='gray')
	# plt.show()

	print(np.argmax(model.predict(img/255)))