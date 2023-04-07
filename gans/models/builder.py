from . import dusty_v1, dusty_v2, vanilla


def build_generator(cfg):
    if cfg.arch == "vanilla":
        G = vanilla.Generator(
            synthesis_kwargs=cfg.synthesis_kwargs,
        )
    elif cfg.arch == "dusty_v1":
        G = dusty_v1.Generator(
            synthesis_kwargs=cfg.synthesis_kwargs,
            measurement_kwargs=cfg.measurement_kwargs,
        )
    elif cfg.arch == "dusty_v2":
        G = dusty_v2.Generator(
            mapping_kwargs=cfg.mapping_kwargs,
            synthesis_kwargs=cfg.synthesis_kwargs,
            measurement_kwargs=cfg.measurement_kwargs,
        )
    else:
        raise ValueError(cfg.arch)
    return G


def build_discriminator(cfg):
    if cfg.arch == "vanilla":
        D = vanilla.Discriminator(**cfg.layer_kwargs)
    elif cfg.arch == "dusty_v2":
        D = dusty_v2.Discriminator(**cfg.layer_kwargs)
    else:
        raise ValueError(cfg.arch)
    return D
