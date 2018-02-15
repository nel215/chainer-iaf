import numpy as np
from iaf import AutoregressiveNN


def test_forward():
    batch_size = 16
    z_size = 32
    h_size = 80
    model = AutoregressiveNN(z_size)
    z = np.random.randn(batch_size, z_size).astype('f')
    h = np.random.randn(batch_size, h_size).astype('f')
    z, loss = model.forward(z, h)
    assert z.shape == (batch_size, z_size)
    assert loss.shape == (batch_size,)
