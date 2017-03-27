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
	
	def fit(self, X_train, y_train):
		
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
