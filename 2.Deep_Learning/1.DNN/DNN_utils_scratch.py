
import numpy as np
import matplotlib.pyplot as plt

#################################################################
#1. Initializer parameters
def initialize_parameters_deep(layer_dims):

	np.random.seed(1)
	parameters = {}
	L = len(layer_dims)

	for l in range(1,L):

		parameters['W'+str(l)] = np.random.randn(layer_dims[l],layer_dims[l-1]) / np.sqrt(layer_dims[l-1])
		parameters['b'+str(l)] = np.zeros((layer_dims[l],1))

	return parameters

#2. linear forward
def linear_forward(A, W, b):

	Z = np.dot(W,A)+b
	cache = (A, W, b)

	return Z, cache

#3. linear activation forward
def linear_activation_forward(A_prev, W, b, activation):

	Z, linear_cache = linear_forward(A_prev, W, b)
	
	if activation == 'relu':
		A = np.maximum(0,Z)
		assert(A.shape == Z.shape)
		activation_cache = Z
	elif activation == 'sigmoid':
		A = 1/(1+np.exp(-Z))
		activation_cache = Z
	elif activation == 'softmax':
		exps = np.exp(Z-np.max(Z, axis=0)) 
		A = exps / np.sum(exps, axis=0)
		activation_cache = Zc

	cache = (linear_cache, activation_cache)

	return A, cache

#4. L model forward
def L_model_forward(X, parameters):

	caches = []
	L = len(parameters) // 2
	A = X

	for l in range(1,L):
		A_prev = A
		A, cache = linear_activation_forward(A_prev, parameters['W'+str(l)], parameters['b'+str(l)], 'relu')
		caches.append(cache)

	AL, cache = linear_activation_forward(A, parameters['W'+str(L)], parameters['b'+str(L)], 'sigmoid')
	caches.append(cache)

	return AL, caches


#5. Compute cost
def compute_cost(AL, Y):

	m = Y.shape[1]
	cost = -1/m*np.sum(np.multiply(Y, np.log(AL))+np.multiply((1-Y),np.log(1-AL)))
	cost = np.squeeze(cost)

	return cost

#----------------------
#6. linear backward
def linear_backward(dZ, cache):

	A_prev, W, b = cache
	m = A_prev.shape[1]

	dW = 1/m*np.dot(dZ, A_prev.T)
	db = 1/m*np.sum(dZ, axis=1, keepdims=True)
	dA_prev = np.dot(W.T, dZ)

	return dA_prev, dW, db

#7. linear_activation_backward
def linear_activation_backward(dA, cache, activation):

	linear_cache, activation_cache = cache
	Z = activation_cache

	if activation == 'relu':
		dZ = np.array(dA, copy=True)
		dZ[Z<=0] = 0
	elif activation == 'sigmoid':
		s = 1/(1+np.exp(-Z))
		dZ = dA*s*(1-s)
	elif activation == 'softmax':
		exps = np.exp(Z-np.max(Z, axis=0)) 
		A = exps / np.sum(exps, axis=0)
		dZ = dA*A*(1-A)

	dA_prev, dW, db = linear_backward(dZ, linear_cache)

	return dA_prev, dW, db

#8. L model backward
def L_model_backward(AL, Y, caches):

	grads = {}
	L = len(caches)
	m = AL.shape[1]
	Y = Y.reshape(AL.shape)

	dAL = -(np.divide(Y,AL)-np.divide(1-Y,1-AL))

	current_cache = caches[L-1]
	grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL, current_cache, 'sigmoid')

	for l in reversed(range(L-1)):
		current_cache = caches[l]
		grads["dA"+str(l)], grads["dW"+str(l+1)], grads["db"+str(l+1)] = linear_activation_backward(grads["dA"+str(l+1)], current_cache, 'relu')

	return grads

#9. Update parameters
def update_parameters(parameters, grads, learning_rate):

	L = len(parameters) // 2

	for l in range(L):
		parameters["W" + str(l+1)] = parameters["W" + str(l+1)]-learning_rate*grads["dW" + str(l+1)]
		parameters["b" + str(l+1)] = parameters["b" + str(l+1)]-learning_rate*grads["db" + str(l+1)]

	return parameters

#10. Predict
def predict(X, y, parameters):
	
	m = X.shape[1]
	n = len(parameters) // 2
	p = np.zeros((1,m))
	# Forward propagation
	probas, caches = L_model_forward(X, parameters)
	# convert probas to 0/1 predictions
	for i in range(0, probas.shape[1]):
		if probas[0,i] > 0.5:
			p[0,i] = 1
		else:
			p[0,i] = 0
	print("Accuracy: "  + str(np.sum((p == y)/m)))

	return p







