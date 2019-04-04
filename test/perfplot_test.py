import numpy
import perfplot


def test():
    kernels = [lambda a: numpy.c_[a, a]]
    r = [2 ** k for k in range(4)]
    perfplot.show(
        setup=numpy.random.rand,
        kernels=kernels,
        labels=["c_"],
        n_range=r,
        xlabel="len(a)",
    )

    perfplot.show(
        setup=numpy.random.rand,
        kernels=kernels,
        labels=["c_"],
        n_range=r,
        xlabel="len(a)",
        logx=True,
        logy=False,
    )

    out = perfplot.bench(
        setup=numpy.random.rand,
        kernels=kernels,
        labels=["c_"],
        n_range=r,
        xlabel="len(a)",
        logx=False,
        logy=True,
    )
    print(out)

    perfplot.show(
        setup=numpy.random.rand,
        kernels=kernels,
        labels=["c_"],
        n_range=r,
        xlabel="len(a)",
        logx=True,
        logy=True,
    )
    return


def test_no_labels():
    def mytest(a):
        return numpy.c_[a, a]

    kernels = [mytest]
    r = [2 ** k for k in range(4)]

    perfplot.plot(setup=numpy.random.rand, kernels=kernels, n_range=r, xlabel="len(a)")
    return


def test_save():
    def mytest(a):
        return numpy.c_[a, a]

    kernels = [mytest]
    r = [2 ** k for k in range(4)]

    perfplot.save(
        "out.png",
        setup=numpy.random.rand,
        kernels=kernels,
        n_range=r,
        xlabel="len(a)",
        title="mytest",
    )
    return
