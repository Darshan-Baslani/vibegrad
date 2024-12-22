from ..core import Tensor
import numpy as np


class Linear:
    def __init__(self, fan_in, fan_out, bias=True):
        self.fan_in = fan_in
        self.fan_out = fan_out
        self.bias_enabled = bias
        self.weight = Tensor(
            np.random.uniform(size=(fan_in, fan_out)) / fan_in**0.5 
        )
        self.bias = Tensor(
            np.zeros(fan_out) if bias else None
        )

    def __repr__(self) -> str:
         return f"\nLinear({self.fan_in}, {self.fan_out}, bias={self.bias_enabled}) "
  
    def __call__(self, x:Tensor):
        out = x @ self.weight
        if self.bias is not None:
                out += self.bias
        return out
  
    def parameters(self):
        return [self.weight] + ([] if self.bias is None else [self.bias])

    def total_params(self):
        return self.fan_in*self.fan_out+self.fan_out if self.bias_enabled else self.fan_in*self.fan_out