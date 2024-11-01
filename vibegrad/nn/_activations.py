from ..core import Tensor
import numpy as np


def relu(tensor: Tensor) -> Tensor:
    out = Tensor(
        np.maximum(0, tensor.data),
        (tensor,),
        "relu"
    )

    def _backward():
        tensor.grad = (tensor.data > 0) * out.grad
    out.grad = _backward
    
    return out

def tanh(tensor: Tensor) -> Tensor:
    x = tensor.data
    t = (np.exp(2*x) - 1)/(np.exp(2*x) + 1)
    out = Tensor(t, (tensor, ), 'tanh')
    
    def _backward():
      tensor.grad += (1 - t**2) * out.grad
    out._backward = _backward
    
    return out