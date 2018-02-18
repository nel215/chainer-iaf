from chainer import Chain
from chainer import links as L
from chainer import functions as F


class FCDecoder(Chain):

    def __init__(self, n_h, n_out):
        super(FCDecoder, self).__init__()
        with self.init_scope():
            self.l0 = L.Linear(None, n_h)
            self.l1 = L.Linear(None, n_out)

    def _forward(self, z):
        h = F.relu(self.l0(z))
        y = self.l1(h)
        return y

    def __call__(self, z):
        y = self._forward(z)
        return y
