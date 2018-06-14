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


def test_fixed_repeat():
    kernels = [lambda a: numpy.c_[a, a]]
    r = [2 ** k for k in range(4)]
    out = perfplot.bench(
        setup=numpy.random.rand,
        kernels=kernels,
        labels=["c_"],
        n_range=r,
        repeat=100,
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


def test_mult_setup():
    def mytest(a):
        return numpy.c_[a, a]
    def mytest2(a):
        return numpy.stack([a, a]).T

    kernels = [mytest, mytest2]

    def setup1(N):
        return numpy.random.rand(N)

    def setup2(N):
        return numpy.random.randint(N)

    setups = [setup1, setup2]
    r = [2 ** k for k in range(4)]

    out = perfplot.bench(
        setup=setups, kernels=kernels, n_range=r, xlabel="len(a)", equality_check=None
    )
    # out.show()
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
