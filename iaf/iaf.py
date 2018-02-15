import math
import numpy as np
from chainer import Chain
from chainer import links as L
from chainer import functions as F


class InverseAutoregressiveFlow(Chain):

    def __init__(self, encoder, ar_models):
        super(InverseAutoregressiveFlow, self).__init__()
        with self.init_scope():
            self.encoder = encoder
            self.ar_models = ar_models

    def forward(self, x):
        batch_size = x.shape[0]
        m, ln_s, h = self.encoder.forward(x)
        z_size = m.shape[1]
        eps = np.random.randn(batch_size, z_size)
        z = F.exp(ln_s) * eps + m
        loss = -F.sum(ln_s + 0.5*eps**2 + 0.5*math.log(2*math.pi), axis=1)
        for ar_model in self.ar_models:
            z, l = ar_model.forward(z, h)
            loss += l

        return z, loss
