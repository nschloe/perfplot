import numpy
import perfplot


def test():
    kernels = [lambda a: numpy.c_[a, a]]
    r = [2 ** k for k in range(4)]
    out = perfplot.bench(
        setup=numpy.random.rand,
        kernels=kernels,
        labels=["c_"],
        n_range=r,
        xlabel="len(a)",
    )
    out.show()

    out = perfplot.bench(
        setup=numpy.random.rand,
        kernels=kernels,
        labels=["c_"],
        n_range=r,
        xlabel="len(a)",
        logx=True,
        logy=False,
    )
    out.show()

    out = perfplot.bench(
        setup=numpy.random.rand,
        kernels=kernels,
        labels=["c_"],
        n_range=r,
        xlabel="len(a)",
        logx=False,
        logy=True,
    )
    out.show()

    out = perfplot.bench(
        setup=numpy.random.rand,
        kernels=kernels,
        labels=["c_"],
        n_range=r,
        xlabel="len(a)",
        logx=True,
        logy=True,
    )
    out.show()
    return


def test_no_labels():
    def mytest(a):
        return numpy.c_[a, a]

    kernels = [mytest]
    r = [2 ** k for k in range(4)]

    out = perfplot.bench(
        setup=numpy.random.rand, kernels=kernels, n_range=r, xlabel="len(a)"
    )
    out.show()
    return


def test_save():
    def mytest(a):
        return numpy.c_[a, a]

    kernels = [mytest]
    r = [2 ** k for k in range(4)]

    out = perfplot.bench(
        setup=numpy.random.rand,
        kernels=kernels,
        n_range=r,
        xlabel="len(a)",
        title="mytest",
    )
    out.save("out.png")
    return
