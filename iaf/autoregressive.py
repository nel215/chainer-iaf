from chainer import Chain
from chainer import links as L
from chainer import functions as F


class AutoregressiveNN(Chain):

    def __init__(self, z_size):
        super(AutoregressiveNN, self).__init__()
        with self.init_scope():
            self.m1 = L.Linear(None, z_size)
            self.m2 = L.Linear(None, z_size)
            self.s1 = L.Linear(None, z_size)
            self.s2 = L.Linear(None, z_size)

    def forward(self, z, h):
        zh = F.concat([z, h])

        m = F.relu(self.m1(zh))
        m = self.m2(m)

        s = F.relu(self.s1(zh))
        s = F.sigmoid(self.s2(s))

        z = s * z + (1.-s) * m
        return z
