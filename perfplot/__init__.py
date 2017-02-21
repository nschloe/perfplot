# -*- coding: utf-8 -*-
#
from perfplot.__about__ import (
        __author__,
        __author_email__,
        __copyright__,
        __license__,
        __version__,
        __status__
        )


def show(setup, kernels, labels, n_range, xlabel=None, repeat=5, number=100):
    from matplotlib import pyplot as plt
    _plot(
        setup, kernels, labels, n_range,
        xlabel=xlabel,
        repeat=repeat,
        number=number
        )
    plt.show()
    return


def _plot(setup, kernels, labels, n_range, xlabel=None, repeat=5, number=100):
    from matplotlib import pyplot as plt
    import numpy
    import timeit

    timings = numpy.empty((len(kernels), len(n_range), repeat))
    for k, kernel in enumerate(kernels):
        for i, n in enumerate(n_range):
            out = setup(n)
            timings[k, i] = timeit.repeat(
                stmt=lambda: kernel(out),
                repeat=repeat,
                number=number
                )
    timings /= number

    # plot minimum
    x = n_range
    T = numpy.min(timings, axis=2)
    for t, label in zip(T, labels):
        plt.plot(x, t, label=label)
    if xlabel:
        plt.xlabel(xlabel)
    plt.ylabel('Time in seconds')
    plt.legend()
    return
