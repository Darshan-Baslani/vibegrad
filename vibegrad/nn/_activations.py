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

def sigmoid(tensor:Tensor) -> Tensor:
    out = Tensor(
        1/(1 + np.exp(-tensor.data)),
        (tensor,),
        "sigmoid"
    )

    def _backward():
        tensor.grad = out.data * (1-out.data) * out.grad
    out._backward = _backward
    
    return out

    
def softmax(tensor:Tensor) -> Tensor:
    # below we are using stablility trick by subtracting the max value
    # this is done to avoid calculations of huge tensors
    exp_logits = np.exp(tensor.data - np.max(tensor.data, keepdims=True))
    out = Tensor(
        exp_logits / np.sum(exp_logits, keepdims=True),
        (tensor,),
        "softmax"
    )

    # TODO: add backward pass
    def _backwar():
        pass
    return out
    