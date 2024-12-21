from ..core import Tensor
from ._activations import *
import numpy as np
from abc import ABC, abstractmethod


class ActivationAbstract(ABC):
    def __init__(self):
        self.param_count = 0
        
    @abstractmethod
    def __call__(self, x:Tensor) -> Tensor:
        pass
    
    @abstractmethod
    def total_params(self) -> int:
        pass
    
    @abstractmethod
    def parameters(self) -> list:
        pass
class ReLU(ActivationAbstract):
    def __init__(self) -> None:
        self.param_count = 0
        
    def __call__(self, x:Tensor) -> Tensor:
        return relu(x)

    def total_params(self) -> int:
        return 0

    def parameters(self) -> list:
        return []

class Sigmoid(ActivationAbstract):
    def __init__(self) -> None:
        self.param_count = 0
    
    def __call__(self, x:Tensor) -> Tensor:
        return sigmoid(x)
    
    def total_params(self) -> int:
        return 0

    def parameters(self) -> list:
        return []