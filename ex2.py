import math
import matplotlib.pyplot as plt
import scipy.optimize as op
import numpy as np
np.set_printoptions(suppress=True)

def plot_cost_history(cost_history):
	plt.clf()
	plt.title('Cost J x iterations')
	
	plt.ylabel('Cost J')
	plt.xlabel('Number of iterations')

	plt.plot(cost_history, 'g.')
	
	plt.show()
	
# 1.2.1 Warmup exercise: sigmoid function
def sigmoid(z):
	if np.isscalar(z):
		return 1 / (1 + math.e ** (-z))
	else:
		return np.ones(z.size) / (np.ones(z.size) + np.repeat(np.e, z.size) ** (-z))
		
def hypothesis(theta, X):
	return sigmoid(np.dot(X, theta))

# 1.2.2 Cost function and gradient
def compute_cost(theta, X, y):
	m = X.shape[0]
	
	h = hypothesis(theta, X)
	error = np.sum(-y * np.log(h) - (1 - y) * np.log(1 - h))

	return error / m
	
# 1.2.3 Learning parameters using fminunc
def gradient(theta, X, y):
	m = X.shape[0]
	
	h = hypothesis(theta, X)
	
	for j in range(theta.shape[0]):
		theta[j] = np.sum((h - y) * X[:, j]) / m
			
	return theta
	
if __name__ == '__main__':
	# Import data
	data = np.loadtxt("ex2data1.txt", delimiter=",")

	positive = data[data[:,2] == 1][:,0:2]
	negative = data[data[:,2] == 0][:,0:2]

	# 1.1 Visualizing the data
	plt.ylabel('Exame 2 score')
	plt.xlabel('Exam 1 score')

	plt.plot(positive[:,0], positive[:,1], 'b+', label = 'Admitted')
	plt.plot(negative[:,0], negative[:,1], 'rx', label = 'Not admitted')

	plt.legend()
	plt.show()

	# 1.2 Implementation
		
	# For large positive values of x, the sigmoid should be close to 1, 
	# while for large negative values, the sigmoid should be close to 0. 
	# Evaluating sigmoid(0) should give you exactly 0.5.
	assert sigmoid(0) == 0.5
	assert sigmoid(129378) == 1
	assert round(sigmoid(-111)) == 0

	# For a matrix, your function should perform the sigmoid function on every element.
	assert np.all(sigmoid(np.array([0,0])) == 0.5)

	theta = np.zeros(3)

	# Evaluation
	## Create the bias column, set to 1
	X = data[:, 0:2]
	X = np.stack((np.ones(data.shape[0]), X[:, 0], X[:, 1]), axis=-1)

	y = data[:, 2]

	# You should see that the cost is about 0.693.
	assert round(compute_cost(theta, X, y), 3) == 0.693

	result = op.minimize(compute_cost, theta, (X, y), 'TNC', gradient, options = {'maxiter': 400})
	theta = result.x
	cost = result.fun

	try:
		# You should see that the cost is about 0.203.
		assert(round(cost, 3) == 0.203)

		# 1.2.4 Evaluating logistic regression
		# You should expect to see an admission probability of 0.776
		assert(round(hypothesis(theta, np.array([1, 45, 85])), 3) == 0.776)
		
	# There is no fminunc in scipy, so I have used https://en.wikipedia.org/wiki/Truncated_Newton_method.
	# However, it does not produce correct theta values, which therefore yelds terrible prediction.
	# In this case, I went back to gradient descent with manually tuned alpha and iterations.
	except:
		theta = np.zeros(3)
		alpha = 0.00417
		iterations = 400000
		cost_history = []

		m = X.shape[0]
		
		for i in range(iterations):
			h = hypothesis(theta, X)
			
			for j in range(theta.shape[0]):
				theta[j] = theta[j] - alpha * np.sum((h - y) * X[:, j]) / m
				
			cost_history.append(compute_cost(theta, X, y))

		cost = compute_cost(theta, X, y)

		# You should see that the cost is about 0.203.
		assert(round(cost, 3) == 0.203)

		# 1.2.4 Evaluating logistic regression
		# You should expect to see an admission probability of 0.776
		assert(round(hypothesis(theta, np.array([1, 45, 85])), 3) == 0.776)
		
		plot_cost_history(cost_history)
		
	# The predict function will produce “1” or “0” predictions given a dataset 
	# and a learned parameter vector θ.
	print('Train Accuracy: ', np.mean((hypothesis(theta, X) >= 0.5).astype(float) == y) * 100)
