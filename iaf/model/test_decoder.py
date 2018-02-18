import numpy as np
from iaf.model import FCDecoder


def test_call():
    n_h = 100
    n_out = 256
    model = FCDecoder(n_h, n_out)

    n_batch = 16
    n_z = 32
    z = np.random.randn(n_batch, n_z).astype('f')
    y = model(z)
    assert y.shape == (n_batch, n_out)
