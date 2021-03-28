import numpy as np

import perfplot


def test_live():
    kernels = [lambda a: np.c_[a, a]]
    r = [2 ** k for k in range(4)]

    perfplot.live(
        setup=np.random.rand,
        kernels=kernels,
        labels=["c_"],
        n_range=r,
        xlabel="len(a)",
    )
