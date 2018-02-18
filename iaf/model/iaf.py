import math
import numpy as np
from chainer import Chain, ChainList
from chainer import links as L
from chainer import functions as F


class IAFBlock(Chain):

    def __init__(self, m_ann, s_ann):
        super(IAFBlock, self).__init__()
        with self.init_scope():
            self.m_ann = m_ann
            self.s_ann = s_ann

    def __call__(self, z, h):
        zh = F.concat([z, h])
        m = self.m_ann(zh)
        s = F.sigmoid(self.s_ann(zh))

        z = s * z + (1.-s) * m
        loss = -F.sum(F.log(s), axis=1)

        return z, loss


class InverseAutoregressiveFlow(ChainList):

    def __call__(self, mu, ln_s, h):
        eps = self.xp.random.standard_normal(mu.shape).astype('f')
        z = F.exp(ln_s) * eps + mu
        loss = -F.sum(ln_s + 0.5*eps**2 + 0.5*math.log(2*math.pi), axis=1)
        for iaf_block in self.children():
            z, l = iaf_block(z, h)
            loss += l

        return z, loss
