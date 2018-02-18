from chainer import Chain
from chainer import functions as F


class VAE(Chain):

    def __init__(self, encoder, iaflow, decoder):
        super(VAE, self).__init__()
        with self.init_scope():
            self.encoder = encoder
            self.iaflow = iaflow
            self.decoder = decoder

    def forward(self, x):
        mu, sigma, h = self.encoder(x)
        z, iaflow_loss = self.iaflow(mu, sigma, h)
        y = self.decoder(z)
        loss = 0
        loss += iaflow_loss
        loss += F.bernoulli_nll(x, y, reduce='no')
        return y, loss

    def __call__(self, x):
        _, loss = self.forward(x)
        return loss
