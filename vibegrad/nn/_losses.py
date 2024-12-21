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

        self.pred = pred
        self.actual = actual

        bce_loss = Tensor(-(actual.data * np.log(pred_clipped) + (1-actual.data) * np.log(1-pred_clipped)))

        if self.reduction == "mean":
            # bce_loss = bce_loss / pred_clipped.shape[0]
            bce_loss = np.mean(bce_loss.data)
        else:
            bce_loss = np.sum(bce_loss.data)

        self.out = Tensor(bce_loss, (pred, actual), "bce_loss")
        return bce_loss