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
    def _initialize_params(self, in_size):
        super(LTLinear, self)._initialize_params(in_size)
        self.triu_indices = np.triu_indices(in_size, m=self.out_size)
        self.W.data[self.triu_indices] = 0

    def __call__(self, x):
        if self.W.data is None:
            in_size = functools.reduce(operator.mul, x.shape[1:], 1)
            self._initialize_params(in_size)

        self.W.data[self.triu_indices] = 0
        return linear.linear(x, self.W, self.b)


class AutoregressiveLinear(Chain):

    def __init__(self, z_size):
        super(AutoregressiveLinear, self).__init__()
        with self.init_scope():
            self.m1 = LTLinear(None, z_size)
            self.m2 = LTLinear(None, z_size)
            self.s1 = LTLinear(None, z_size)
            self.s2 = LTLinear(None, z_size)

    def forward(self, z, h):
        zh = F.concat([z, h])

        m = F.relu(self.m1(zh))
        m = self.m2(m)

        s = F.relu(self.s1(zh))
        s = F.sigmoid(self.s2(s))

        z = s * z + (1.-s) * m
        loss = -F.sum(s, axis=1)
        return z, loss
