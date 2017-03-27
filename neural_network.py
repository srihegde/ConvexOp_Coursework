import numpy as np
import sklearn
import sklearn.datasets
import matplotlib.pyplot as plt

# Note : current support only for
# Activation - {'tanh'}
# Solver - {'sgd'}

class NNClassifier:
	def __init__(self, activation = 'tanh', solver = 'sgd', hidden_layer_sizes = (100,), alpha = 0.0001, batch_size = 'auto', learning_rate_init = 0.001):   
		self.activation = activation
		self.solver = solver
		self.hidden_layer_sizes = hidden_layer_sizes
		self.alpha = alpha
		self.batch_size = batch_size
		self.learning_rate_init = learning_rate_init
		self.model = {}
		self.model['parameters'] = {}
		self.model['parameters']['W'] = {}
		self.model['parameters']['b'] = {}
		self.output_label_map = {}

	def act_derivative(self, output):
		if (self.activation == 'sigmoid'):
			return output * (1.0 - output)
		elif self.activation == 'tanh':
			return 1 - output*output

	def forwardPass(self, sample, W_list, b_list):
		a,z = np.array(sample),0
		output = list()
		for layer_no in xrange(self.no_of_layers-1):
			z = a.dot(W_list[layer_no]) + b_list[layer_no]
			a = np.tanh(z)
			output.append(a)

		exp_score = np.exp(z)
		probs = exp_score / np.sum(exp_score, axis=1, keepdims=True)
		output.append(probs)

		return output

	def backwardPass(self, expected, output, deltaList, W_list, b_list):
		tmpdel = list()
		for k in reversed(range(self.no_of_layers)):

			errors = np.array(list())
			if k != self.no_of_layers-1:
				# errors = (W_list[k] + b_list[k]) * tmpdel[self.no_of_layers-k-2]
				print W_list[k].shape, b_list[k].shape, np.array(tmpdel[self.no_of_layers-k-2]).shape
				errors = (b_list[k].T * tmpdel[self.no_of_layers-k-2])# + (W_list[k].T * tmpdel[self.no_of_layers-k-2])
			else:
				errors = output[k] - expected

			tmpdel.append(errors * self.act_derivative(output[k]))
		return tmpdel


	def updateWeights(self, sample,output, deltaList, W_list, b_list):
		for i in range(self.no_of_layers):
			inputs = sample
			if i != 0:
				inputs = output[i-1]

			W_list[i] = self.learning_rate_init*deltaList[i]*inputs
			b_list[i] = self.learning_rate_init*deltaList[i]

		return (W_list,b_list)
		

	def trainSGD(self, W_list, b_list, X_train, y_train, num_passes):
		
		deltaList=list()
		for i in xrange(num_passes):
			for j in xrange(len(X_train)):

				output = self.forwardPass(X_train[j], W_list, b_list)
				expected = np.array([0 for i in range(self.output_layer_dim)])
				expected[(y_train[j]+1)%self.output_layer_dim] = 1
				deltaList = self.backwardPass(expected, output, deltaList, W_list, b_list)
				(W_list, b_list) = self.updateWeights(X_train[j], output, deltaList, W_list, b_list)
	
		return (W_list,b_list)


	def fit(self, X_train, y_train, num_passes=20):
		
		assert(len(X_train) == len(y_train) and len(X_train) > 0)
			
		#Initialize model parameters
	
		self.no_of_layers = len(self.hidden_layer_sizes) + 2
		self.input_layer_dim = len(X_train[0])
		self.output_layer_dim = len( set(y_train) )
		
		#TO-DO 
		#Actually verify this label mapping after implementing gradient descent
		for idx, label in enumerate( set(y_train) ):
			self.output_label_map[idx] = label 

		layer_sizes = [self.input_layer_dim] + list( self.hidden_layer_sizes ) + [self.output_layer_dim] 

		W_list, b_list = [], []
	
		for layer_no in xrange(1, self.no_of_layers):
			W_list.append( np.random.randn(layer_sizes[layer_no-1], layer_sizes[layer_no]) / np.sqrt(layer_sizes[layer_no-1]) )
			b_list.append( np.zeros((1,layer_sizes[layer_no])) )
		
		#Train here using gradient descent
		
		(W_list, b_list) = self.trainSGD(W_list, b_list, X_train, y_train, num_passes)
		

		#Update self.model with final parameters
		
		for idx, W_idx in enumerate(W_list):
			self.model['parameters']['W'][idx] = W_idx
		for idx, b_idx in enumerate(b_list):
			self.model['parameters']['b'][idx] = b_idx
	
	#Input - X_test : the input data to be classified ( shape = (n_samples,n_features) )
	#Output - y_test : output labels ( shape = (n_samples) )
	def predict(self, X_test):
		
		a = np.array( X_test )
		
		for layer_no in xrange(self.no_of_layers-1):
			z = a.dot(self.model['parameters']['W'][layer_no]) + self.model['parameters']['b'][layer_no]
			a = np.tanh(z) 
		
		exp_score = np.exp(z)
		probs = exp_score / np.sum(exp_score, axis=1, keepdims=True)

    		#print probs
	
		raw_labels = np.argmax(probs, axis=1)
		return [ self.output_label_map[label] for label in raw_labels ]
