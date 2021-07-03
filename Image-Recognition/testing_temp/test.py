from data import get_mnist
import numpy as np
import matplotlib.pyplot as plt


"""
w = weights, b = bias, i = input, h = hidden, o = output, l = label
e.g. w_i_h = weights from input layer to hidden layer
"""

data = np.load('data/mnist.npz')
lst = data.files
for item in lst:
    print(item)
    print(data[item])

images, labels = get_mnist()
w_i_h = np.random.uniform(-0.5, 0.5, (20, 784))
w_h_o = np.random.uniform(-0.5, 0.5, (10, 20))
b_i_h = np.zeros((20, 1))

m, n = w_i_h.shape


print(m)

print(n)
