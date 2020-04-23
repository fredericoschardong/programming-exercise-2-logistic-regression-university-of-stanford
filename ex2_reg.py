import math
import matplotlib.pyplot as plt
import scipy.optimize as op
import numpy as np
np.set_printoptions(suppress=True)

from ex2 import hypothesis, plot_cost_history

def plot_classification(positive, negative, theta, title):
	#from https://towardsdatascience.com/andrew-ngs-machine-learning-course-in-python-regularized-logistic-regression-lasso-regression-721f311130fb
	def mapFeaturePlot(x1,x2,degree):
		"""
		take in numpy array of x1 and x2, return all polynomial terms up to the given degree
		"""
		out = np.ones(1)
		for i in range(1,degree+1):
			for j in range(i+1):
				terms= (x1**(i-j) * x2**j)
				out= np.hstack((out,terms))
		return out
		
	plt.clf()
	plt.title(title)
	plt.plot(positive[:,0], positive[:,1], 'b+', label = 'y = 1')
	plt.plot(negative[:,0], negative[:,1], 'rx', label = 'y = 0')

	# Plotting decision boundary
	u_vals = np.linspace(-1,1.5,50)
	v_vals = np.linspace(-1,1.5,50)
	z = np.zeros((len(u_vals),len(v_vals)))

	for i in range(len(u_vals)):
		for j in range(len(v_vals)):
			z[i,j] = mapFeaturePlot(u_vals[i],v_vals[j],6) @ theta
			
	plt.contour(u_vals,v_vals,z.T,0)
	plt.ylabel('Microchip Test 2')
	plt.xlabel('Microchip Test 1')
	plt.legend()
	plt.show()

# Import data
data = np.loadtxt("ex2data2.txt", delimiter=",")

positive = data[data[:,2] == 1][:,0:2]
negative = data[data[:,2] == 0][:,0:2]

# 2.1 Visualizing the data
plt.ylabel('Microchip Test 2')
plt.xlabel('Microchip Test 1')

plt.plot(positive[:,0], positive[:,1], 'b+', label = 'y = 1')
plt.plot(negative[:,0], negative[:,1], 'rx', label = 'y = 0')

plt.legend()
plt.show()

# 2.2 Feature mapping
def map_feature(x1, x2):
	degree = 6
	out = np.ones(x1.shape[0])
	
	for i in range(1, degree + 1):
		for j in range(i + 1):
			out = np.c_[out, (x1 ** (i - j)) * (x2 ** j)]
			
	return out

X = map_feature(data[:, 0], data[:, 1])

# 2.3 Cost function and gradient
def cost_function_reg(theta, X, y, Lambda):
	m = X.shape[0]
	
	h = hypothesis(theta, X)
	error = np.sum(-y * np.log(h) - (1 - y) * np.log(1 - h))

	error = error / m
	
	return error + ((Lambda / (2*m)) * np.sum(theta ** 2))
	
# Evaluation	
theta = np.zeros(X.shape[1])
y = data[:, 2]

# Set regularization parameter lambda to 1
Lambda = 1
 
# You should see that the cost is about 0.693.
assert round(cost_function_reg(theta, X, y, Lambda), 3) == 0.693

def gradient(theta, X, y, Lambda):
	m = X.shape[0]
	
	h = hypothesis(theta, X)
	
	# for j = 0
	theta[0] = np.sum((h - y) * X[:, 0]) / m
	
	# for j ≥ 1
	for j in range(1, theta.shape[0]):
		theta[j] = (np.sum((h - y) * X[:, j]) / m) + ((Lambda / m) * theta[j])
			
	return theta
	
# There is no fminunc in scipy, so I have used gradient descent with manually tuned alpha and iterations.
for iterations in [100, 1000]:
	for Lambda in [0, 0.5]:
		theta = np.zeros(X.shape[1])
		alpha = 0.1
		cost_history = []

		m = X.shape[0]

		for i in range(iterations):
			m = X.shape[0]

			h = hypothesis(theta, X)
			
			# for j = 0
			theta[0] = theta[0] - alpha * np.sum((h - y) * X[:, 0]) / m
			
			# for j ≥ 1
			for j in range(1, theta.shape[0]):
				theta[j] = theta[j] - alpha * (np.sum((h - y) * X[:, j]) / m) + ((Lambda / m) * theta[j])
				
			cost_history.append(cost_function_reg(theta, X, y, Lambda))
			
		#plot_cost_history(cost_history)
		cost = cost_function_reg(theta, X, y, Lambda)
		
		# The predict function will produce “1” or “0” predictions given a dataset 
		# and a learned parameter vector θ.
		accuracy = np.mean((hypothesis(theta, X) >= 0.5).astype(float) == y) * 100

		result = 'Iterations = %d, lambda = %.1f, Accuracy = %.2f%%' % (iterations, Lambda, accuracy)
		
		print(result)

		#plot_classification(positive, negative, theta, result)
