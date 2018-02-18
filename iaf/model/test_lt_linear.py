import numpy as np
from numpy.testing import assert_almost_equal
from iaf.model.autoregressive import LTLinear


def test_call():
    batch_size = 4
    x_size = 16
    z_size = 8
    x = np.random.randn(batch_size, x_size).astype('f')
    lt_linear = LTLinear(None, z_size)
    lt_linear(x)
    triu_indices = np.triu_indices(x_size, m=z_size)
    assert_almost_equal(lt_linear.W.data[triu_indices], 0)
