import chainer
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
        z, iaf_loss = self.iaflow(mu, ln_s, h)
        y = self.decoder(z)
        iaf_loss = F.sum(iaf_loss) / x.shape[0]
        rec_loss = F.bernoulli_nll(x, y) / x.shape[0]
        loss = 0.1 * iaf_loss + rec_loss

        chainer.report(
            {'loss': loss, 'iaf_loss': iaf_loss, 'rec_loss': rec_loss},
            observer=self,
        )
        return y, loss

    def __call__(self, x):
        _, loss = self.forward(x)
        return loss

    def generate(self, x):
        with chainer.using_config('train', False), chainer.no_backprop_mode():
            y, _ = self.forward(x)
            return F.sigmoid(y).data
