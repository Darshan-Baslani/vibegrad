from ..core import Tensor
import numpy as np
from ._optims import Optims


class SGD(Optims):
    def __init__(self, model_parameters:list, lr:float):
        self.model_parameters = model_parameters
        self.lr = lr

    def step(self):
        for p in self.model_parameters:
            if p is not None:
                p.data.setflags(write=True)
                p.data -= self.lr * p.grad

    def zero_grad(self):
        for p in self.model_parameters:
            if isinstance(p, Tensor):
                p.grad = np.zeros_like(p.data)