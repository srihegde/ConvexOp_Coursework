import mnist
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pickle

from neural_network import NNClassifier 

np.random.seed(42)

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
	np.random.shuffle(imgs_n)
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
	(train_data, train_labels) = prepareData('training', 1, 500)
	# Preparing testing data
	(test_data, test_labels) = prepareData('testing', 1, 100)

	# print len(train_data), len(train_labels), len(test_data), len(test_data)

	# Initialize Neural Network and predict
	# clf = NNClassifier(hidden_layer_sizes = (5,), learning_rate_init = 0.05, num_passes = 200) 	
	# clf.fit(train_data, train_labels)

	# pickle.dump(clf, open("model_1_500.p","wb"))

	clf = pickle.load( open("model_1_500.p", "rb") )

	result = clf.predict(test_data)

	#Results

	tp = 0
	fp = 0
	correct = 0
	for i in xrange(len(test_labels)):
		if(test_labels[i] == 1 and result[i] == 1): tp += 1
		if(test_labels[i] == 1 and result[i] == 0): fp += 1 
		if(test_labels[i] == result[i]): correct += 1

	print result.count(1), result.count(0)

	print "Accuracy : ", float(correct)/len(test_labels)
	print "Precision : ", float(tp)/result.count(1)
	print "Recall : ", float(tp)/(fp + tp)
