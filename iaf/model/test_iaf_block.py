import numpy as np
from iaf.model import IAFBlock, AutoregressiveLinear


def test_call():
    n_z = 32
    m_ann = AutoregressiveLinear(n_z)
    s_ann = AutoregressiveLinear(n_z)
    model = IAFBlock(m_ann, s_ann)

    n_batch = 16
    n_h = 80
    z = np.random.randn(n_batch, n_z).astype('f')
    h = np.random.randn(n_batch, n_h).astype('f')
    z, loss = model(z, h)
    assert z.shape == (n_batch, n_z)
    assert loss.shape == (n_batch,)
