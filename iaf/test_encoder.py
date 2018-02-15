import numpy as np
from iaf import Encoder


def test_forward():
    batch_size = 16
    x_size = 256
    z_size = 32
    h_size = 80
    model = Encoder(z_size, h_size)
    x = np.random.randn(batch_size, x_size).astype('f')
    m, s, h = model.forward(x)
    assert m.shape == (batch_size, z_size)
    assert s.shape == (batch_size, z_size)
    assert h.shape == (batch_size, h_size)
