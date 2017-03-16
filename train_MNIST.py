import mnist
import cv2
import matplotlib.pyplot as plt


def showImg(img):
	plt.imshow(img, cmap='Greys_r')
	plt.show()


if __name__ == '__main__':

	# Preparing training data
	imgs, labs = mnist.load_mnist('training', digits = [3])
	# showImg(imgs[0])