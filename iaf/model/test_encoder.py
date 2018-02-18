import numpy as np
from iaf.model import FCEncoder


def test_call():
    n_z = 32
    model = FCEncoder(n_z, 100)

    n_batch = 16
    n_x = 256
    x = np.random.randn(n_batch, n_x).astype('f')
    y = model(x)
    assert y.shape == (n_batch, n_z)
