from ..core import Tensor
from ._activations import *
import numpy as np


class ReLU:
    def __init__(self) -> None:
        pass

    def __call__(self, x:Tensor) -> Tensor:
        return relu(x)

    def total_params(self) -> int:
        return 0

class Sigmoid:
    def __init__(self) -> None:
        pass
    
    def __call__(self, x:Tensor) -> Tensor:
        return sigmoid(x)
    
    def total_params(self) -> int:
        return 0