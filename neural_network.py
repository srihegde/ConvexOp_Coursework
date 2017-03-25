import numpy as np
import sklearn
import sklearn.datasets
import matplotlib.pyplot as plt

class NNClassifier:
	def __init__(self, activation = 'tanh', solver = 'sgd', hidden_layers_sizes = (100,), alpha = 0.0001, batch_size = 'auto', learning_rate_init = 0.001)   
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
	
	def fit(self, X_train, y_train):
		
		assert(len(X_train) == len(y_train) and len(X_train) > 0)
			
		#Initialize model parameters
	
		no_of_layers = len(hidden_layer_sizes) + 2
		input_layer_dim = len(X_train[0])
		output_layer_dim = len( set(y_train) )

		layer_sizes = [input_layer_dim] + self.hidden_layer_sizes + [output_layer_dim] 

		W_list, b_list = [], []
	
		for layer_no in xrange(1,no_of_layers):
			W_list.append( np.random.randn(layer_sizes[layer_no-1], layer_sizes[layer_no]) / np.sqrt(layer_sizes[layer_no-1]) )
			b_list.append( np.zeros((1,layer_sizes[layer_no])) )
		
		#Train here using gradient descent

		
		
		#Update self.model with final parameters
		
		for idx, W_idx in enumerate(W_list):
			self.model['parameters']['W'][idx] = W_idx
		for idx, b_idx in enumerate(b_list):
			self.model['parameters']['b'][idx] = b_idx

		return self
		 
