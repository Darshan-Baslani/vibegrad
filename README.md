# VibeGrad: A Lightweight Automatic Differentiation Library

## Overview
![Local Image](https://github.com/Darshan-Baslani/vibegrad/blob/main/vibegrad.png "Local Image")
MicroGrad is a minimal, educational automatic differentiation library implemented from scratch in Python. It provides a computational graph-based approach to computing gradients, making it perfect for building and understanding neural networks from the ground up.

## Features

- *Automatic Differentiation*: Vibegrad automatically computes gradients, making it easy to implement backpropagation.

- *Tensor Operations*: Supports basic tensor operations necessary for building and training models.

- *Optimization*: Includes common optimization algorithms like Stochastic Gradient Descent (SGD).

- *Loss Functions*: Provides essential loss functions such as Binary Cross-Entropy Loss (BCELoss).

## Quick Start
```python
from vibegrad import Tensor

# Create tensors
x = Tensor(3.0, requires_grad=True)
y = Tensor(2.0, requires_grad=True)

# Perform operations
z = x * y + x**2
z.backward()

# Get gradients
print(x.grad)  # Expected gradient value
print(y.grad)  # Expected gradient value
```

## Computational Graph

The library constructs a computational graph dynamically as operations are performed, allowing for efficient backpropagation of gradients.
