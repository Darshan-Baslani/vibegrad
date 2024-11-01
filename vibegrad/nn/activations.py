from ..core import Tensor
from ._activations import *
import numpy as np


class ReLU:
    def __init__(self) -> None:
        pass

    def __call__(self) -> Tensor:
        return relu(self)

    def total_params(self):
        return 0