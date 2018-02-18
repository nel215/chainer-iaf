from chainer import Chain
from chainer import functions as F
from iaf.model import FCEncoder


class VAE(Chain):

    def __init__(self, n_z, n_h, iaflow, decoder):
        super(VAE, self).__init__()
        with self.init_scope():
            self.mu_encoder = FCEncoder(n_z, 100)
            self.sigma_encoder = FCEncoder(n_z, 100)
            self.h_encoder = FCEncoder(n_h, 100)
            self.iaflow = iaflow
            self.decoder = decoder

    def forward(self, x):
        mu = self.mu_encoder(x)
        ln_s = self.sigma_encoder(x)
        h = self.h_encoder(x)
        z, iaflow_loss = self.iaflow(mu, ln_s, h)
        y = self.decoder(z)
        loss = 0
        loss += iaflow_loss
        loss += F.bernoulli_nll(x, y, reduce='no')
        return y, loss

    def __call__(self, x):
        _, loss = self.forward(x)
        return loss
