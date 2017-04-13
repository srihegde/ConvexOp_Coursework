import numpy as np
import math
from random import randint

# Class for defining the SVM functionalities
class svm():

	def __init__(self, C = 1.0, kernel = 'none', gsigma = 1.0, tol = 1e-3, max_passes = 1000):
		self.C = C
		self.kernel = kernel
		self.gsigma = gsigma
		self.tol = tol
		self.max_passes = max_passes



	# Kernel function mapping for inner product (if any)
	def kernel_fn(self,x,y):
		val = 0
		if self.kernel == 'gaussian':
			val = math.exp(np.linalg.norm(x-y)/(-2*(self.gsigma**2)))
		else:
			val = np.inner(x,y)

		return val



	# Implements the Sequential Minimal Optimization(SMO) algorithm
	def SMO_naive(self, x_train, y_train):
		alphas, b = np.zeros(len(x_train)), 0.
		passes = 0
		while passes < self.max_passes:
			
			changed_alphas = 0
			for i in xrange(len(x_train)):
				f_x = 0
				for j in xrange(len(x_train)):
					f_x += ((alphas[j] * y_train[j] * self.kernel_fn(x_train[j], x_train[i])) + b)
				E_i = f_x - y_train[i]

				if (E_i*y_train[i] < -tol and alphas[i] < self.C) or (E_i*y_train[i] > tol and alphas[i] > 0):
					j = randint(0, len(x_train)-2)
					if j == i:
						j = len(x_train)-1
					
					f_x = 0
					for k in xrange(len(x_train)):
						f_x += ((alphas[k] * y_train[k] * self.kernel_fn(x_train[k], x_train[j])) + b)
					
					E_j = f_x - y_train[j]
					alpha_i_old, alpha_j_old = alphas[i], alphas[j]

					L,H = 0,0
					if y_train[i] == y_train[j]:
						L = max(0,alphas[i] - alphas[j])
						H = min(self.C, self.C + alphas[j] - alphas[i])
					else:
						L = max(0,alphas[i] + alphas[j] - self.C)
						H = min(self.C, alphas[j] + alphas[i])

					eta = 2*self.kernel_fn(x_train[i], x_train[j]) - self.kernel_fn(x_train[i],x_train[i]) - self.kernel_fn(x_train[j],x_train[j])
					if L == H or eta >= 0:
						continue

					alphas[j] -= (y_train[j]*(E_i - E_j)/eta)
					if alphas[j] > H: alphas[j] = H
					elif alphas[j] < L: alphas[j] = L

					if math.fabs(alphas[j] - alpha_j_old) < 1e-5:
						continue

					alphas[i] += y_train[i]*y_test[j]*(alpha_j_old - alphas[j])
					b1 = b - E_i - (y_train[i]*(alphas[i] - alpha_i_old)*self.kernel_fn(x_train[i], x_train[i])) - (y_train[j]*(alphas[j] - alpha_j_old)*self.kernel_fn(x_train[i], x_train[j]))
					b2 = b - E_j - (y_train[i]*(alphas[i] - alpha_i_old)*self.kernel_fn(x_train[i], x_train[j])) - (y_train[j]*(alphas[j] - alpha_j_old)*self.kernel_fn(x_train[j], x_train[j]))

					if alphas[i] > 0 and alphas[i] < self.C:
						b = b1
					elif alphas[j] > 0 and alphas[j] < self.C:
						b = b2
					else:
						b = (b1+b2)/2

					changed_alphas += 1

			if changed_alphas == 0:
				passes += 1
			else:
				passes = 0

		return (alphas, b)



	# Fits the SVM model according to the given training data
	def fit(self, x_train, y_train):
		pass

	# Performs classification on samples in X
	def predict(self, x_test):
		pass

	# Returns the mean accuracy on the given test data and labels
	def score(self, y_test, y_pred):
		tp = 0
		fp = 0
		correct = 0
		for i in xrange(len(y_test)):
			if(y_test[i] == 1 and y_pred[i] == 1): tp += 1
			if(y_test[i] == 1 and y_pred[i] == 0): fp += 1 
			if(y_test[i] == y_pred[i]): correct += 1

		acc = float(correct)/len(y_test)
		precision = float(tp)/result.count(1)
		recall = float(tp)/(fp + tp)

		return (acc, precision, recall)