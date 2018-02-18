import numpy as np
from iaf.model import AutoregressiveLinear


def test_call():
    n_batch = 16
    n_out = 32
    model = AutoregressiveLinear(n_out)

    n_x = 256
    x = np.random.randn(n_batch, n_x).astype('f')
    y = model(x)
    assert y.shape == (n_batch, n_out)
