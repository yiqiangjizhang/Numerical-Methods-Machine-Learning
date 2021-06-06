# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# # Image recognition using Neural Network
#
# The following notebook is implemented a two-layer neural network for image recognition using Python.
#
# The training set used is the MNIST dataset.

# %%
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

# Libraries
import numpy as np  # Numpy library for numerical operations and linear algebra
# Pandas library for data science tools and data processing, CSV file I/O (e.g. pd.read_csv)
import pandas as pd
from matplotlib import pyplot as plt  # Matplotlib library for MATLAB tools

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All"
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# %% [markdown]
# ### Read data

# %%
# Read data from data set
data = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')

# %% [markdown]
# ### Preview data

# %%
data.head


# %%
# Transform data to array
data = np.array(data)
# Get the number of rows 'm' and columns 'n'
m_original, n_original = data.shape

# Shuffle data before splitting
np.random.shuffle(data)

# %% [markdown]
# ### Split data into test and training set

# %%
# Spit data from training and test
nTest = 1000

# Test set
data_test = data[0:nTest].T
Y_test = data_test[0]
X_test = data_test[1:n_original]
X_test = X_test / 255.

# Train set
data_train = data[nTest:m_original].T
Y_train = data_train[0]
X_train = data_train[1:n_original]
X_train = X_train / 255.
_, m_train = X_train.shape

# _,m_train = X_train.shape

# Y_train = Y_train / 255. # Normalize (to avoid exp overflow)


# %%
Y_train

# %% [markdown]
# The NN will feature a straightforward two-layer design. The input layer $a[0]$ will have 784 units, which match to the 784 pixels in each 28x28 input picture. A hidden layer $a[1]$ will have 10 units with ReLU activation, and the output layer $a[2]$ will have 10 units with softmax activation corresponding to the ten digit classes.
#
# **Forward propagation**
#
# $$Z^{[1]} = W^{[1]} X + b^{[1]}$$
# $$A^{[1]} = g_{\text{ReLU}}(Z^{[1]}))$$
# $$Z^{[2]} = W^{[2]} A^{[1]} + b^{[2]}$$
# $$A^{[2]} = g_{\text{softmax}}(Z^{[2]})$$
#
# **Backward propagation**
#
# $$dZ^{[2]} = A^{[2]} - Y$$
# $$dW^{[2]} = \frac{1}{m} dZ^{[2]} A^{[1]T}$$
# $$dB^{[2]} = \frac{1}{m} \Sigma {dZ^{[2]}}$$
# $$dZ^{[1]} = W^{[2]T} dZ^{[2]} .* g^{[1]\prime} (z^{[1]})$$
# $$dW^{[1]} = \frac{1}{m} dZ^{[1]} A^{[0]T}$$
# $$dB^{[1]} = \frac{1}{m} \Sigma {dZ^{[1]}}$$
#
# **Parameter updates**
#
# $$W^{[2]} := W^{[2]} - \alpha dW^{[2]}$$
# $$b^{[2]} := b^{[2]} - \alpha db^{[2]}$$
# $$W^{[1]} := W^{[1]} - \alpha dW^{[1]}$$
# $$b^{[1]} := b^{[1]} - \alpha db^{[1]}$$
#
# **Vars and shapes**
#
# Forward prop
#
# - $A^{[0]} = X$: 784 x m
# - $Z^{[1]} \sim A^{[1]}$: 10 x m
# - $W^{[1]}$: 10 x 784 (as $W^{[1]} A^{[0]} \sim Z^{[1]}$)
# - $B^{[1]}$: 10 x 1
# - $Z^{[2]} \sim A^{[2]}$: 10 x m
# - $W^{[1]}$: 10 x 10 (as $W^{[2]} A^{[1]} \sim Z^{[2]}$)
# - $B^{[2]}$: 10 x 1
#
# Backprop
#
# - $dZ^{[2]}$: 10 x m ($~A^{[2]}$)
# - $dW^{[2]}$: 10 x 10
# - $dB^{[2]}$: 10 x 1
# - $dZ^{[1]}$: 10 x m ($~A^{[1]}$)
# - $dW^{[1]}$: 10 x 10
# - $dB^{[1]}$: 10 x 1
# %% [markdown]
# ### Functions
# Below are the list of functions that will be used

# %%
# Inititate parameters


def initateParameters():
    # Get arbitrary weights and biases
    # Substract 0.5 since randn generate values from 0 to 1
    W1 = np.random.rand(10, 784) - 0.5  # Matrix of 70 x 784
    b1 = np.random.rand(10, 1) - 0.5  # Vector of 10 x 1
    W2 = np.random.rand(10, 10) - 0.5  # Matrix of 10 x 10
    b2 = np.random.rand(10, 1) - 0.5  # Vector of 10 x 1
    return W1, b1, W2, b2

# ReLU activation function


def ReLU(Z):
    return np.maximum(Z, 0)

# ReLU derivative activation function


def ReLU_deriv(Z):
    return Z > 0

# Softmax activation function


def softmax(Z):
    A = np.exp(Z) / sum(np.exp(Z))
    return A

# Sigmoid activation function
# def Sigmoid(Z):
#    S = 1 / (1 + np.exp(-Z))
#    return S

# Sigmoid derivative activation function
# def Sigmoid_deriv(Z):
#    S_deriv = Sigmoid(Z) * (1 - Sigmoid(Z))
#    return S_deriv

# Forward propagation


def forwardPropagation(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2

# Complete Y data, return the activation


def oneHot(Y):
    # Since there are 0 - 9 numbers = 10
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y

# Backward propagation


def backwardPropagation(Z1, A1, Z2, A2, W1, W2, X, Y):
    one_hot_Y = oneHot(Y)
    dZ2 = A2 - one_hot_Y
    dW2 = 1 / m_original * dZ2.dot(A1.T)
    db2 = 1 / m_original * np.sum(dZ2)
    dZ1 = W2.T.dot(dZ2) * ReLU_deriv(Z1)
    dW1 = 1 / m_original * dZ1.dot(X.T)
    db1 = 1 / m_original * np.sum(dZ1)
    return dW1, db1, dW2, db2

# Update parameters


def updateParameters(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1
    W2 = W2 - alpha * dW2
    b2 = b2 - alpha * db2
    return W1, b1, W2, b2


# %%
# Return prediction
def getPredictions(A2):
    # Return the index of the maximum argunment, thus, the predicted number index of the [10 x 1] output vector
    return np.argmax(A2, 0)

# Prediction accuracy


def getAccuracy(predictions, Y):
    print(predictions, Y)
    # Accuracy of the predicted number
    return np.sum(predictions == Y) / Y.size

# Gradient descent function


def gradientDescent(X, Y, alpha, iterations):
    W1, b1, W2, b2 = initateParameters()
    # Loop through the amount of iterations we set
    for i in range(iterations):
        Z1, A1, Z2, A2 = forwardPropagation(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = backwardPropagation(Z1, A1, Z2, A2, W1, W2, X, Y)
        W1, b1, W2, b2 = updateParameters(
            W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
        # For every iterations, print prediction
        if i % 100 == 0:
            print("Iteration: ", i)
            predictions = getPredictions(A2)
            print(getAccuracy(predictions, Y))
    return W1, b1, W2, b2

# %% [markdown]
# ### Execute code


# %%
W1, b1, W2, b2 = gradientDescent(X_train, Y_train, 0.10, 1000)

# %% [markdown]
# ### Test an image

# %%
# To make a singular prediction with the weights and biases calculated


def makePredictions(X, W1, b1, W2, b2):
    _, _, _, A2 = forwardPropagation(W1, b1, W2, b2, X)
    predictions = getPredictions(A2)
    return predictions

# Test prediction


def testPredictions(index, W1, b1, W2, b2):
    current_image = X_train[:, index, None]
    prediction = makePredictions(X_train[:, index, None], W1, b1, W2, b2)
    label = Y_train[index]
    print("Prediction: ", prediction)
    print("Label: ", label)

    current_image = current_image.reshape((28, 28)) * 255
    plt.gray()
    plt.imshow(current_image, interpolation='nearest')
    plt.show()

# %% [markdown]
# ### Check some examples of the train set


# %%
testPredictions(0, W1, b1, W2, b2)
testPredictions(1000, W1, b1, W2, b2)
testPredictions(40123, W1, b1, W2, b2)
testPredictions(40000, W1, b1, W2, b2)

# %% [markdown]
# ### Check for the test set

# %%
test_set_predictions = makePredictions(X_test, W1, b1, W2, b2)
getAccuracy(test_set_predictions, Y_test)
