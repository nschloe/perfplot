import numpy
import perfplot


def test():
    kernels = [lambda a: numpy.c_[a, a]]
    r = [2**k for k in range(4)]
    perfplot.show(
            setup=lambda n: numpy.random.rand(n),
            kernels=kernels, labels=['c_'], n_range=r, xlabel='len(a)',
            correctness_check=lambda a, b: (a == b).all()
            )
    perfplot.show(
            setup=lambda n: numpy.random.rand(n),
            kernels=kernels, labels=['c_'], n_range=r, xlabel='len(a)',
            logx=True, logy=False,
            correctness_check=lambda a, b: (a == b).all()
            )
    perfplot.show(
            setup=lambda n: numpy.random.rand(n),
            kernels=kernels, labels=['c_'], n_range=r, xlabel='len(a)',
            logx=False, logy=True,
            correctness_check=lambda a, b: (a == b).all()
            )
    perfplot.show(
            setup=lambda n: numpy.random.rand(n),
            kernels=kernels, labels=['c_'], n_range=r, xlabel='len(a)',
            logx=True, logy=True,
            correctness_check=lambda a, b: (a == b).all()
            )
    return
