import mnist
import cv2
import matplotlib.pyplot as plt
import numpy as np

from random import shuffle
from neural_network import NNClassifier 

def showImg(img):
	plt.imshow(img, cmap='Greys_r')
	plt.show()

def prepareData(option, dig, size):
	
	imgs, labs = mnist.load_mnist(option, digits = [dig])
	imgs_p = imgs[:size]

	imgs_n = []
	for i in xrange(10):
		if i != dig:
			imgs, labs1 = mnist.load_mnist(option, digits = [i])
			imgs_n.extend(imgs[:(size/8)])

	# Shuffle negative samples
	shuffle(imgs_n)
	imgs_n = imgs_n[:size]

	labels = [1 for i in xrange(size)]
	labels.extend([0 for i in xrange(size)])


	# Scaling the data between [0,1]
	imgs_p = [np.divide(np.array(i,dtype=float),255.0) for i in imgs_p]
	imgs_n = [np.divide(np.array(i,dtype=float),255.0) for i in imgs_n]

	# Flatten into feature vector
	fimp = [i.flatten().tolist() for i in imgs_p]
	fimn = [i.flatten().tolist() for i in imgs_n]
	data = fimp
	data.extend(fimn)

	return (data, labels)

if __name__ == '__main__':

	# Preparing training data
	(train_data, train_labels) = prepareData('training', 3, 300)
	# Preparing testing data
	(test_data, test_labels) = prepareData('testing', 3, 100)

	# print len(train_data), len(train_labels), len(test_data), len(test_data)

	# Initialize Neural Network and predict
	clf = NNClassifier(hidden_layer_sizes = (5,)) 	
	clf.fit(train_data, train_labels)
	result = clf.predict(test_data)

	print result.count(1), result.count(0)
