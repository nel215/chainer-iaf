import numpy as np
from iaf import AutoregressiveLinear
from iaf.model import InverseAutoregressiveFlow


def test_call():
    n_z = 32
    n_h = 80
    ar_models = [
        AutoregressiveLinear(n_z)
    ]
    model = InverseAutoregressiveFlow(ar_models)

    n_batch = 16
    mu = np.random.randn(n_batch, n_z).astype('f')
    ln_s = np.random.randn(n_batch, n_z).astype('f')
    h = np.random.randn(n_batch, n_h).astype('f')
    z, loss = model(mu, ln_s, h)
    assert z.shape == (n_batch, n_z)
    assert loss.shape == (n_batch,)
