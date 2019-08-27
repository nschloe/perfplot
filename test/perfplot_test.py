import numpy
import perfplot

kernels = [lambda a: numpy.c_[a, a]]
r = [2 ** k for k in range(4)]


def test():
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
    perfplot.plot(
        setup=numpy.random.rand,
        kernels=kernels,
        n_range=r,
        xlabel="len(a)"
    )
    return


def test_automatic_scale():
    from perfplot.main import PerfplotData

    # (expected_prefix, time in nanoseconds, expected_timing) format
    test_cases = [
        ('ns', 0.125, 0.125),  # Dealing w/ edge-case when timing < nanosecond
        ('us', 9.999e5, 999.9),  # Almost a milisecond
        ('ms', 1e6, 1.0),  # Equal exactly to a milisecond
        ('s', 1.5e9, 1.5),  # Over 1 second
        ('s', 1e9, 1.0)  # Tests if disabling ``automatic_scale`` yields seconds
    ]
    for i, (exp_prefix, time_ns, exp_timing) in enumerate(test_cases):
        data = PerfplotData(
            n_range=[1],
            # Converting timings to ns
            timings=numpy.full((1, 1), time_ns, dtype=numpy.float64),
            labels=[""],
            colors=[""],
            xlabel="",
            title="",
            logx=False,
            logy=False,
            automatic_order=True,
            # True except for last test-case
            automatic_scale=(True if i != len(test_cases) - 1 else False)
        )
        # Has the correct prefix been applied?
        assert data.timings_unit == exp_prefix

        # Have the timings been updated correctly?
        assert numpy.min(data.timings) == exp_timing


def test_save():
    perfplot.save(
        "out.png",
        setup=numpy.random.rand,
        kernels=kernels,
        n_range=r,
        xlabel="len(a)",
        title="mytest",
    )
    return
