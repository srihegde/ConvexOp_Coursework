import mnist
import cv2
import matplotlib.pyplot as plt
import numpy as np
 

def showImg(img):
	plt.imshow(img, cmap='Greys_r')
	plt.show()


if __name__ == '__main__':

	# Preparing training data
	imgs, labs = mnist.load_mnist('training', digits = [3])
	imgs_3 = imgs[:2000]
	imgs, labs1 = mnist.load_mnist('training', digits = [8])
	imgs_8 = imgs[:2000]
	train_labels = [-1 for i in xrange(2000)]
	train_labels.extend([1 for i in xrange(2000)])


	# Scaling the data between [0,1]
	imgs_3 = [np.divide(np.array(i,dtype=float),255.0) for i in imgs_3]
	imgs_8 = [np.divide(np.array(i,dtype=float),255.0) for i in imgs_8]

	# Flatten into feature vector
	fim3 = [i.flatten().tolist() for i in imgs_3]
	fim8 = [i.flatten().tolist() for i in imgs_8]
	train_data = fim3
	train_data.extend(fim8)


	# Preparing testing data
	imgs, labs = mnist.load_mnist('testing', digits = [3])
	imgs_3 = imgs[:500]
	imgs, labs1 = mnist.load_mnist('testing', digits = [8])
	imgs_8 = imgs[:500]
	test_labels = [-1 for i in xrange(500)]
	test_labels.extend([1 for i in xrange(500)])

	# Scaling the data between [0,1]
	imgs_3 = [np.divide(np.array(i,dtype=float),255.0) for i in imgs_3]
	imgs_8 = [np.divide(np.array(i,dtype=float),255.0) for i in imgs_8]

	# Flatten into feature vector
	fim3 = [i.flatten().tolist() for i in imgs_3]
	fim8 = [i.flatten().tolist() for i in imgs_8]
	test_data = fim3
	test_data.extend(fim8)

