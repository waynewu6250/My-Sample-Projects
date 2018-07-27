import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from mnist import load_mnist
from DNN_utils_scratch import *


#Load data
(x_train, y_train), (x_test, y_test) = load_mnist(flatten=True, normalize=False)

# Show sample
sample1 = x_train[1,:].reshape(28, 28)
label1 = y_train[1]
print('Label:', label1)

# pil_img = Image.fromarray(np.uint8(sample1))
# pil_img.show()
# plt.imshow(sample1)
# plt.show()
x_train = x_train.T
x_test = x_test.T
y_train=np.eye(10)[y_train.reshape(-1)].T
y_test=np.eye(10)[y_test.reshape(-1)].T

print('X_train shape:', x_train.shape)
print('Y_train shape:', y_train.shape)
print('X_test shape:', x_test.shape)
print('Y_test shape:', y_test.shape)

x_train_1 = x_train[:,:10000]
x_test_1 = x_test[:,:10000]
y_train_1 = y_train[:,:10000]
y_test_1 = y_test[:,:10000]

#---------------------
layers_dims = [784, 20, 7, 5, 10]

def L_layer_model(X, Y, layers_dims, learning_rate = 0.001, num_iterations = 500):

	np.random.seed(1)
	costs = []
	iteration = []

	parameters = initialize_parameters_deep(layers_dims)

	for i in range(0, num_iterations):

		AL, caches = L_model_forward(X, parameters)
		cost = compute_cost(AL, Y)
		grads = L_model_backward(AL, Y, caches)
		parameters = update_parameters(parameters, grads, learning_rate)

		if i % 100 ==0:
			print("Cost after iterations %i: %f" % (i, cost))
			costs.append(cost)
			iteration.append(i)


	plt.plot(iteration,np.squeeze(costs))
	plt.ylabel('cost')
	plt.xlabel('iterations (per tens)')
	plt.title('Learning rate = ' + str(learning_rate))
	plt.show()

	return parameters


#-----------------------
parameters = L_layer_model(x_train, y_train, layers_dims)

pred_train = predict(x_train, y_train, parameters)

pred_test = predict(x_test, y_test, parameters)
