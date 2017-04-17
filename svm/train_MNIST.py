import mnist
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pickle

import svm

np.random.seed(42)

def showImg(img):
	plt.imshow(img, cmap='Greys_r')
	plt.show()

def prepareData(option, dig, size):
	
	imgs, labs = mnist.load_mnist(option, digits = [dig])
	np.random.shuffle(imgs)
	imgs_p = imgs[:size]

	imgs_n = []
	for i in xrange(10):
		if i != dig:
			imgs, labs1 = mnist.load_mnist(option, digits = [i])
			imgs_n.extend(imgs[:(size/5)])

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
	(train_data, train_labels) = prepareData('training', 1, 200)
	# Preparing testing data
	(test_data, test_labels) = prepareData('testing', 1, 200)

	if(len(test_data) != len(test_labels)):
		print 'Changing data size to ', len(test_data)
		test_labels = test_labels[:len(test_data)]

	print 'Training Data size:', len(train_data) 
	print 'Testing Data size:', len(test_data)

	# Initialize svm and predict
	# clf = svm.SVM()
	# clf.fit(train_data, train_labels)
	# pickle.dump(clf, open("./models/model_1_400.p","wb"))
	clf = pickle.load( open("./models/model_1_400.p", "rb") )

	# Results
	test_pred = clf.predict(test_data)
	(acc, prec, recall) = clf.score(test_labels, test_pred)
	print 'Accuracy = ',acc
	print 'Precision = ',prec
	print 'Recall = ',recall