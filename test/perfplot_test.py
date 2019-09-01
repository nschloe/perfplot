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
    perfplot.plot(setup=numpy.random.rand, kernels=kernels, n_range=r, xlabel="len(a)")
    return


def test_automatic_scale():
    from perfplot.main import PerfplotData
    from matplotlib import pyplot as plt
    import re

    # Regular Expression that retrieves the plot unit from label
    unit_re = re.compile(r"\[([mun]?[s])\]")

    # (expected_unit, time in nanoseconds, expected_timing, time_unit) format
    test_cases = [
        # Dealing w/ edge-case when timing < nanosecond
        ("ns", 0.125, None),
        # Almost a milisecond
        ("us", 9.999e5, None),
        # Equal exactly to a milisecond
        ("ms", 1e6, None),
        # Over 1 second
        ("s", 1.5e9, None),
        # Checking if providing 's' for time_unit yields seconds
        ("s", 1e9, "s"),
    ]

    for exp_unit, time_ns, time_unit in test_cases:
        data = PerfplotData(
            n_range=[1],
            # Converting timings to ns
            timings=numpy.full((1, 1), time_ns, dtype=numpy.uint64),
            labels=["."],  # Suppress no handle error # TODO fix this
            colors=None,
            xlabel="",
            title="",
            logx=False,
            logy=False,
            automatic_order=True,
            time_unit=time_unit,
        )
        # Has the correct unit been selected?
        assert data.time_unit == exp_unit

        # Has the correct unit been applied to the y_label?
        data.plot()
        ax = plt.gca()
        plot_unit = unit_re.search(ax.get_ylabel()).groups()[0]
        assert plot_unit == exp_unit


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
