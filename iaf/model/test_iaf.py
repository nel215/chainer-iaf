import numpy as np
from iaf.model import InverseAutoregressiveFlow, IAFBlock, AutoregressiveLinear


def test_call():
    n_z = 32
    n_h = 80
    m_ann = AutoregressiveLinear(n_z)
    s_ann = AutoregressiveLinear(n_z)
    ar_models = [
        IAFBlock(m_ann, s_ann),
    ]
    model = InverseAutoregressiveFlow(*ar_models)

    n_batch = 16
    mu = np.random.randn(n_batch, n_z).astype('f')
    ln_s = np.random.randn(n_batch, n_z).astype('f')
    h = np.random.randn(n_batch, n_h).astype('f')
    z, loss = model(mu, ln_s, h)
    assert z.shape == (n_batch, n_z)
    assert loss.shape == (n_batch,)
