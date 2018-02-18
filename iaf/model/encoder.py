from chainer import Chain
from chainer import links as L
from chainer import functions as F


class FCEncoder(Chain):

    def __init__(self, n_z, n_h):
        super(FCEncoder, self).__init__()
        with self.init_scope():
            self.l0 = L.Linear(None, n_h)
            self.l1 = L.Linear(None, n_z)

    def __call__(self, x):
        h = F.relu(self.l0(x))
        y = self.l1(h)

        return y
