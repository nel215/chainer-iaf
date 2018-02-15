import numpy as np
from iaf import InverseAutoregressiveFlow, Encoder, AutoregressiveNN


def test_forward():
    z_size = 32
    h_size = 80
    encoder = Encoder(z_size, h_size)
    ar_models = [
        AutoregressiveNN(z_size)
    ]
    model = InverseAutoregressiveFlow(encoder, ar_models)

    batch_size = 16
    x_size = 256
    x = np.random.randn(batch_size, x_size).astype('f')
    z, loss = model.forward(x)
    assert z.shape == (batch_size, z_size)
    assert loss.shape == (batch_size,)
