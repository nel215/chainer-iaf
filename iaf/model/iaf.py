import math
import numpy as np
from chainer import Chain
from chainer import links as L
from chainer import functions as F


class InverseAutoregressiveFlow(Chain):

    def __init__(self, ar_models):
        super(InverseAutoregressiveFlow, self).__init__()
        with self.init_scope():
            self.ar_models = ar_models

    def __call__(self, mu, ln_s, h):
        eps = self.xp.random.standard_normal(mu.shape).astype('f')
        z = F.exp(ln_s) * eps + mu
        loss = -F.sum(ln_s + 0.5*eps**2 + 0.5*math.log(2*math.pi), axis=1)
        for ar_model in self.ar_models:
            z, l = ar_model.forward(z, h)
            loss += l

        return z, loss
