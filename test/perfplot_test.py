import numpy
import perfplot


def test():
    setup = lambda n: numpy.random.rand(n)
    kernels = [lambda a: numpy.c_[a, a]]
    r = [2**k for k in range(4)]
    perfplot.show(setup, kernels, labels=['c_'], n_range=r, xlabel='len(a)')
    perfplot.show(
            setup, kernels, labels=['c_'], n_range=r, xlabel='len(a)',
            logx=True, logy=False
            )
    perfplot.show(
            setup, kernels, labels=['c_'], n_range=r, xlabel='len(a)',
            logx=False, logy=True
            )
    perfplot.show(
            setup, kernels, labels=['c_'], n_range=r, xlabel='len(a)',
            logx=True, logy=True
            )
    return
