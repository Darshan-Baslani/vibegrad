import numpy as np

class Tensor:
    def __init__(self, data, _children=(), _op='', label=''):
        self.data = np.array(data, dtype=np.float64) if not isinstance(data, np.ndarray) else data
        self.grad = np.zeros_like(self.data, dtype=np.float64)
        self._prev = set(_children)
        self._op = _op
        self.label = label
        self._backward = lambda : None
        self.shape = self.data.shape    

    def __repr__(self) -> str:
        return f"Tensor(data={self.data})"

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data+other.data, (self, other), '+')

        def _backward():
            if self.shape != other.shape:
                other.data = np.broadcast_to(other.data, self.shape)
                other.grad = np.broadcast_to(other.grad, self.grad.shape)
            self.grad += out.grad
            other.grad.setflags(write=True)
            other.grad += out.grad
        out._backward = _backward
        
        return out

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        return self + (-other)

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data*other.data, (self,other), "*")

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward

        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers"
        out = Tensor(self.data**other, (self,), f"**{other}")

        def _backward():
            self.grad += (other * self.data**(other-1)) * out.grad
        out._backward = _backward

        return out

    def relu(self):
        out = Tensor(0 if self.data < 0 else self.data, (self,), 'ReLU')

        def _backward():
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward

        return out

    def backward(self):

        # topological order all of the children in the graph
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        # go one variable at a time and apply the chain rule to get its gradient
        self.grad = 1
        for v in reversed(topo):
            v._backward()

    def __neg__(self): # -self
        return self * -1

    def __radd__(self, other): # other + self
        return self + other

    def __sub__(self, other): # self - other
        return self + (-other)

    def __rsub__(self, other): # other - self
        return other + (-self)

    def __rmul__(self, other): # other * self
        return self * other

    def __truediv__(self, other): # self / other
        return self * other**-1

    def __rtruediv__(self, other): # other / self
        return other * self**-1

    def __matmul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(
            np.matmul(self.data, other.data),
            (self, other),
            "@"
        )

        def _backward():
            # For matrix multiplication C = A @ B:
            # dL/dA = dL/dC @ B.T
            # dL/dB = A.T @ dL/dC
            self.grad = np.matmul(out.grad, other.data.T)
            other.grad = np.matmul(self.data.T, out.grad)
        out.backward = _backward
        return out

    def __rmatmul__(self, other):
        return Tensor(other) @ self

    def zero_grad(self):
        self.grad = float(0.0)
        
        