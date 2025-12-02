# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
#     path: /usr/share/jupyter/kernels/python3
# ---

# %% [markdown]
# # Neural networks from scratch
#
# ## Libraries and dataset

# %%
import math

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

n_classes = 3
n_loops = 1
n_samples = 1500

def spirals(n_classes=3, n_samples=1500, n_loops=2):
    klass = np.random.choice(n_classes, n_samples)
    radius = np.random.rand(n_samples)
    theta = klass * 2 * math.pi / n_classes + radius * 2 * math.pi * n_loops
    radius = radius + 0.05 * np.random.randn(n_samples)
    return np.column_stack((radius * np.cos(theta), radius * np.sin(theta))).astype("float32"), klass

X, y = spirals(n_samples=n_samples, n_classes=n_classes, n_loops=n_loops)

# %% [markdown]
# ## Visualize the dataset

# %%
...


# %% [markdown]
# ## Activation functions
#
# ReLU and sigmoid function and their derivative (should work for numpy
# array of any dimension (1D, 2D,…))

# %%
def relu(v):
    ...


def drelu(v):
    ...


def sigmoid(v):
    ...


def dsigmoid(v):
    ...


# %% [markdown]
# ## Defining the neural network
#
# First define the shape of the neural network:
#
# -   `n0`: size of input,
# -   `n1`: size of hidden layer,
# -   `n2`: size of output.

# %%
n0 = ...
n1 = ...
n2 = ...

# %% [markdown]
# Variables for weights, biases of each layers and intermediate variables
# to compute the gradient.

# %%
# Random weights
W1 = np.random.randn(n0, n1)
W2 = np.random.randn(n1, n2)

# Biases set to zero
b1 = np.zeros(n1)
b2 = np.zeros(n2)

# Partial derivatives of output w.r.t. activations, see slide
# "Backpropagation equations"
Xx_1 = np.zeros((n2, n1))
Xx_2 = np.zeros((n2, n2))

# Partial derivatives of output w.r.t. biases, see slide
# "Backpropagation equations"
Xb_1 = np.zeros((n2, n1))
Xb_2 = np.zeros((n2, n2))

# Partial derivatives of output w.r.t. weights, see slide
# "Backpropagation equations"
Xw_1 = np.zeros((n2, n0, n1))
Xw_2 = np.zeros((n2, n1, n2))

# Partial derivatives of loss w.r.t. weights and biases, see slide
# "Cross entropy loss"
Lw_1 = np.zeros((n0, n1))
Lw_2 = np.zeros((n1, n2))
Lb_1 = np.zeros(n1)
Lb_2 = np.zeros(n2)

# %% [markdown]
# Define the learning rate and the activation functions along their
# derivatives at each layer:
#
# -   `eta`: learning rate
# -   `af`, `daf`: activation function and its derivative for hidden layer

# %%
# Define eta, af, daf
eta = ...
af = ...
daf = ...

# %% [markdown]
# ## The learning loop

# %%
nepochs = 15
for epoch in range(nepochs + 1):
    acc_epoch = 0
    for idx, (x0, y2) in enumerate(zip(X, y)):
        # Implement the forward pass: use `W1`, `x0`, `b1`, `af`, `W2`, `x1`,
        # `b2` to define `z1`, `x1`, `z2`. Instead of a second activation
        # function, we will use a softmax transformation to define `x2`.
        z1 = ...
        x1 = ...
        z2 = ...
        x2 = ...

        # Predicted class
        pred = np.argmax(x2)
        acc_epoch += (pred == y2)

        if idx % 100 == 0:
            print(f"Epoch: {epoch:02}, sample: {idx:04}, class: {y2}, pred: {pred}, prob: {x2}")

        # To initialize the recurrent relation (3), see slide
        # "Backpropagation equations"
        Xx_2 = ...

        # Update partial derivatives of output w.r.t. weights and biases on
        # second layer. Pay attention, it is the last layer so there is no
        # activation function.
        for i in range(n2):
            for p in range(n2):
                # See equation (2) in slide "Backpropagation equations"
                Xb_2[i, p] = ...
                for q in range(n1):
                    # See equation (1) in slide "Backpropagation equations"
                    Xw_2[i, q, p] = ...

        # Update partial derivatives of output w.r.t. output of hidden layer
        for i in range(n2):
            for p in range(n1):
                Xx_1[i, p] = 0
                for j in range(n2):
                    # See equation (3) in slide "Backpropagation equations"
                    Xx_1[i, p] += ...

        # Update partial derivatives of output w.r.t. weights and
        # biases of hidden layer
        for i in range(n2):
            for p in range(n1):
                # See equation (2) in slide "Backpropagation equations"
                Xb_1[i, p] = ...
                for q in range(n0):
                    # See equation (1) in slide "Backpropagation equations"
                    Xw_1[i, q, p] = ...

        # One-hot encoding of class `y2`
        y2_one_hot = np.zeros(n2)
        y2_one_hot[y2] = 1

        # Compute partial derivatives of the loss w.r.t weights and
        # biases.
        for p in range(n2):
            for q in range(n1):
                Lw_2[q, p] = 0
                for i in range(n2):
                    # Partial derivatives of cross-entropy loss w.r.t.
                    # weights, see slide "Cross-entropy loss"
                    Lw_2[q, p] += ...

        for p in range(n1):
            for q in range(n0):
                Lw_1[q, p] = 0
                for i in range(n2):
                    # Partial derivatives of cross-entropy loss w.r.t.
                    # weights, see slide "Cross-entropy loss"
                    Lw_1[q, p] += ...

        for p in range(n2):
            Lb_2[p] = 0
            for i in range(n2):
                # Partial derivatives of cross-entropy loss w.r.t.
                # biases, see slide "Cross-entropy loss"
                Lb_2[p] += ...

        for p in range(n1):
            Lb_1[p] = 0
            for i in range(n2):
                # Partial derivatives of cross-entropy loss w.r.t.
                # biases, see slide "Cross-entropy loss"
                Lb_1[p] += ...

        # Gradient descent step: use `eta`, `Lw_1` `Lw_2` `Lb_1` `Lb_2` to
        # update `W1`, `W2`, `b1`, `b2`.
        W1 -= ...
        W2 -= ...
        b1 -= ...
        b2 -= ...

    print(f"Epoch: {epoch:02}, training accuracy: {acc_epoch/n_samples}")

# %% [markdown]
# ## Visualization

# %%
num = 250
xx = np.linspace(X[:, 0].min(), X[:, 0].max(), num)
yy = np.linspace(X[:, 1].min(), X[:, 1].max(), num)
XX, YY = np.meshgrid(xx, yy)
points = np.c_[XX.ravel(), YY.ravel()]

# Forward pass on all points
z1 = W1.T @ points.T + b1[:, np.newaxis]
x1 = af(z1)
z2 = W2.T @ x1 + b2[:, np.newaxis]
x2_hat = np.argmax(z2, axis=0)

C = x2_hat.reshape(num, num)

cm_bright = ListedColormap(['#FF0000', '#0000FF', '#00FF00'])
plt.contourf(XX, YY, C, cmap=cm_bright, alpha=.2)
plt.scatter(*X.T, c=y, cmap=cm_bright)

plt.show()
