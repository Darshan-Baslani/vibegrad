from ..core import Tensor
import numpy as np

def _check_binary(pred:Tensor, actual:Tensor):
    for pr in pred.data:
        for p in pr:
            if p != 1 and p != 0:
                raise ValueError("Values must be 0 or 1")

class BCELoss:
    def __init__(self, reduction:str = "mean"):
        """
        Binary Cross Entropy Loss

        Args:
            reduction (str, optional): "mean" or "sum". Defaults to "mean".
        """
        self.reduction = reduction

    def __call__(self, pred:Tensor, actual:Tensor) -> Tensor:
        # _check_binary(pred, actual)
        
        
        epsilon = 1e-15
        pred_clipped = np.clip(pred.data, epsilon, 1-epsilon)

        bce_loss = -(actual.data * np.log(pred_clipped) + (1-actual.data) * np.log(1-pred_clipped))

        if self.reduction == "mean":
            # bce_loss = bce_loss / pred_clipped.shape[0]
            bce_loss = np.mean(bce_loss.data)
        elif self.reduction == 'sum':
            bce_loss = np.sum(bce_loss.data)
        else:
            raise ValueError("Invalid reduction type. Use 'mean' or 'sum'")

        loss = Tensor(bce_loss, (pred, actual), "bce_loss") 
        def _backward():
            grad_pred = -(actual.data / pred_clipped - (1 - actual.data) / (1 - pred_clipped))
            pred.grad += grad_pred * loss.grad
        loss._backward = _backward

        return loss
        