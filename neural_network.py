import numpy as np
import sklearn
import sklearn.datasets
import matplotlib.pyplot as plt

np.random.seed(42)
# Note : current support only for
# Activation - {'tanh'}
# Solver - {'sgd'}

class NNClassifier:
	def __init__(self, activation = 'tanh', solver = 'sgd', hidden_layer_sizes = (100,), alpha = 0.0001, batch_size = 'auto', learning_rate_init = 0.05, num_passes = 50):   
		self.activation = activation
		self.solver = solver
		self.hidden_layer_sizes = hidden_layer_sizes
		self.alpha = alpha
		self.batch_size = batch_size
		self.learning_rate_init = learning_rate_init
		self.num_passes = num_passes
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
			output.append(np.transpose(a))

		#exp_score = np.exp(z)
		#probs = exp_score / np.sum(exp_score, axis=1, keepdims=True)
		return output

	def backwardPass(self, expected, output, W_list, b_list):
		
		deltaList = list()
		
		for k in reversed(range(self.no_of_layers-1)):

			if k != self.no_of_layers-2:
				errors = []
				for neuron_no in xrange(len(output[k])):
					error = 0
					for neuron in xrange(len(output[k+1])):
						error += W_list[k+1][neuron_no,neuron] * deltaList[-1][neuron] 
					errors.append(error)
				errors = np.array(errors)
			else:
				errors = expected - output[k] 
				#print "ERRZ", errors

			deltaList.append(errors * self.act_derivative(output[k]))

		return deltaList[::-1] #Reverse it

	def updateWeights(self, sample, output, deltaList, W_list, b_list):

		for i in range(self.no_of_layers-1):
			inputs = sample
			if i != 0:
				inputs = output[i-1]

			for inp_no, inp in enumerate(inputs): 
				W_list[i][inp_no,:] += self.learning_rate_init*deltaList[i]*inp
			b_list[i] += self.learning_rate_init*deltaList[i]

		return (W_list,b_list)
		

	def trainSGD(self, W_list, b_list, X_train, y_train, num_passes):
		
		for i in xrange(num_passes):
			for j in xrange(len(X_train)):

				output = self.forwardPass(X_train[j], W_list, b_list)

				expected = np.array([0 for i in range(self.output_layer_dim)])
				expected[y_train[j]] = 1
			
				deltaList = self.backwardPass(expected, output, W_list, b_list)

				(W_list, b_list) = self.updateWeights(X_train[j], output, deltaList, W_list, b_list)
				
		return (W_list,b_list)


	def fit(self, X_train, y_train):
		
		assert(len(X_train) == len(y_train) and len(X_train) > 0)
			
		#Initialize model parameters
	
		self.no_of_layers = len(self.hidden_layer_sizes) + 2
		self.input_layer_dim = len(X_train[0])
		self.output_layer_dim = len( set(y_train) )
		
		# #TO-DO 
		# #Actually verify this label mapping after implementing gradient descent
		# for idx, label in enumerate( set(y_train) ):
		# 	self.output_label_map[idx] = label 

		layer_sizes = [self.input_layer_dim] + list( self.hidden_layer_sizes ) + [self.output_layer_dim] 

		W_list, b_list = [], []
	
		for layer_no in xrange(1, self.no_of_layers):
			W_list.append( np.random.randn(layer_sizes[layer_no-1], layer_sizes[layer_no]) / np.sqrt(layer_sizes[layer_no-1]) )
			b_list.append( np.zeros((layer_sizes[layer_no])) )
	
		for i,w in enumerate(W_list): print "Layer ", i,i+1, " parameters shape :", w.shape
		for i,b in enumerate(b_list) : print "Layer ", i,i+1, " bias shape :", b.shape
		print

		#Train here using gradient descent		
		(W_list, b_list) = self.trainSGD(W_list, b_list, X_train, y_train, self.num_passes)
		
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
		
		raw_labels = np.argmax(a, axis=1)
		return list(raw_labels)
		#return [ self.output_label_map[label] for label in raw_labels ]
