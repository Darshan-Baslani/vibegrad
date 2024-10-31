from ..core import Tensor
import numpy as np


class Sequential:
    def __init__(self, layers):
        self.layers = layers
        self.parameters = self._get_params()

    def __call__(self, X):
        for layer in self.layers:
            X = layer(X)
        self.out = X
        return self.out

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def total_params(self):
        total_param = 0
        for layer in self.layers:
            total_param += layer.total_params()
            # for p in layer.parameters():
                # total_param += p.data
                
        return total_param

    def _get_params(self):
        out_str = ""
        for layer in self.layers:
            out_str += layer.__repr__()
        return out_str