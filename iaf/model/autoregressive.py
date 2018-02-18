import numpy as np
from chainer import Chain
from chainer import links as L
from chainer import functions as F
from chainer.functions.connection import linear
from chainer.links.connection.linear import functools, operator


class LTLinear(L.Linear):
    """
    Lower Triangle Linear
    """
    def __init__(self, *args, **kwargs):
        super(LTLinear, self).__init__(*args, **kwargs)
        self.add_persistent('triu_indices', None)

    def _initialize_params(self, in_size):
        super(LTLinear, self)._initialize_params(in_size)
        self.triu_indices = np.array(np.triu_indices(in_size, m=self.out_size))
        self.W.data[self.triu_indices] = 0

    def __call__(self, x):
        if self.W.data is None:
            in_size = functools.reduce(operator.mul, x.shape[1:], 1)
            self._initialize_params(in_size)

        self.W.data[self.triu_indices] = 0
        return linear.linear(x, self.W, self.b)


class AutoregressiveLinear(Chain):

    def __init__(self, n_out):
        super(AutoregressiveLinear, self).__init__()
        with self.init_scope():
            self.l0 = LTLinear(None, n_out)
            self.l1 = LTLinear(None, n_out)

    def __call__(self, x):
        h = F.relu(self.l0(x))
        y = self.l1(h)
        return y
