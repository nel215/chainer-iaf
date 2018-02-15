from chainer import Chain
from chainer import links as L
from chainer import functions as F


class Encoder(Chain):

    def __init__(self, z_size, h_size):
        super(Encoder, self).__init__()
        with self.init_scope():
            self.m1 = L.Linear(None, z_size)
            self.m2 = L.Linear(None, z_size)
            self.s1 = L.Linear(None, z_size)
            self.s2 = L.Linear(None, z_size)
            self.h1 = L.Linear(None, h_size)
            self.h2 = L.Linear(None, h_size)

    def forward(self, x):
        m = F.relu(self.m1(x))
        m = self.m2(m)

        s = F.relu(self.s1(x))
        s = self.s2(s)

        h = F.relu(self.h1(x))
        h = self.h2(h)

        return m, s, h
