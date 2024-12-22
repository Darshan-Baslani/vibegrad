from ..core import Tensor
from abc import ABC, abstractmethod


class Optims(ABC):
    @abstractmethod
    def __init__(self):
        pass
    
    @abstractmethod
    def step(self):
        pass
    
    @abstractmethod
    def zero_grad(self):
        pass