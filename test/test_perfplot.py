import numpy as np
import pytest

import perfplot

kernels = [lambda a: np.c_[a, a]]
r = [2 ** k for k in range(4)]


def test():
    perfplot.show(
        setup=np.random.rand,
        kernels=kernels,
        labels=["c_"],
        n_range=r,
        xlabel="len(a)",
    )

    perfplot.show(
        setup=np.random.rand,
        kernels=kernels,
        labels=["c_"],
        n_range=r,
        xlabel="len(a)",
        logx=True,
        logy=False,
    )

    out = perfplot.bench(
        setup=np.random.rand,
        kernels=kernels,
        labels=["c_"],
        n_range=r,
        xlabel="len(a)",
    )
    print(out)

    perfplot.show(
        setup=np.random.rand,
        kernels=kernels,
        labels=["c_"],
        n_range=r,
        xlabel="len(a)",
        logx=True,
        logy=True,
    )


def test_no_labels():
    perfplot.plot(setup=np.random.rand, kernels=kernels, n_range=r, xlabel="len(a)")


# (expected_unit, time in nanoseconds, expected_timing, time_unit) format
@pytest.mark.parametrize(
    "exp_unit, time_ns, time_unit",
    [
        # Dealing w/ edge-case when timing < nanosecond
        # ("ns", 0.125, "auto"),
        # Almost a milisecond
        ("us", 9.999e5, "auto"),
        # Equal exactly to a milisecond
        ("ms", 1e6, "auto"),
        # Over 1 second
        ("s", 1.5e9, "auto"),
        # Checking if providing 's' for time_unit yields seconds
        ("s", 1e9, "s"),
    ],
)
def test_automatic_scale(exp_unit, time_ns, time_unit):
    import re

    import matplotlib.pyplot as plt

    from perfplot._main import PerfplotData

    timings = np.full((1, 1), time_ns * 1.0e-9)

    data = PerfplotData(
        n_range=[1],
        # Converting timings to ns
        timings_s=timings,
        labels=["."],  # Suppress no handle error # TODO fix this
        xlabel="",
        flop=None,
    )
    # Has the correct unit been applied to the y_label?
    data.plot(time_unit=time_unit)
    ax = plt.gca()

    # Regular Expression that retrieves the plot unit from label
    unit_re = re.compile(r"\[([musn]?[s])\]")
    plot_unit = unit_re.search(ax.get_ylabel()).groups()[0]
    assert plot_unit == exp_unit


def test_save():
    perfplot.save(
        "out.png",
        setup=np.random.rand,
        kernels=kernels,
        n_range=r,
        xlabel="len(a)",
        relative_to=0,
    )


def test_flops():
    perfplot.show(
        setup=np.random.rand,
        kernels=kernels,
        labels=["c_"],
        n_range=r,
        xlabel="len(a)",
        flops=lambda n: n,
    )
