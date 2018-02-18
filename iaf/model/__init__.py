from iaf.model.encoder import FCEncoder  # noqa
from iaf.model.decoder import FCDecoder  # noqa
from iaf.model.iaf import InverseAutoregressiveFlow, IAFBlock  # noqa
from iaf.model.autoregressive import AutoregressiveLinear  # noqa
from iaf.model.vae import VAE  # noqa


def create_sample_model(n_x):
    n_z = 100
    n_h = 200
    m_ann = AutoregressiveLinear(n_z)
    s_ann = AutoregressiveLinear(n_z)
    iaflow = InverseAutoregressiveFlow(*[
        IAFBlock(m_ann, s_ann),
    ])
    decoder = FCDecoder(n_h, n_x)
    model = VAE(n_z, n_h, iaflow, decoder)

    return model
