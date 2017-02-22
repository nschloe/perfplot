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

import pipdated
if pipdated.needs_checking('perfplot'):
    msg = pipdated.check('perfplot', __version__)
    if msg:
        print(msg)


def show(
        setup, kernels, labels, n_range,
        xlabel=None,
        repeat=5,
        number=100,
        logx=False,
        logy=False,
        automatic_order=True
        ):
    from matplotlib import pyplot as plt
    _plot(
        setup, kernels, labels, n_range,
        xlabel=xlabel,
        repeat=repeat,
        number=number,
        logx=logx,
        logy=logy,
        automatic_order=automatic_order
        )
    plt.show()
    return


def _plot(
        setup, kernels, labels, n_range,
        xlabel=None,
        repeat=5,
        number=100,
        logx=False,
        logy=False,
        automatic_order=True,
        ):
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

    # choose plot function
    if logx and logy:
        plotfun = plt.loglog
    elif logx:
        plotfun = plt.semilogx
    elif logy:
        plotfun = plt.semilogy
    else:
        plotfun = plt.plot
    # plot minimum time
    x = n_range
    T = numpy.min(timings, axis=2)

    if automatic_order:
        # Sort T by the last entry. This makes the order in the legend
        # correspond to the order of the lines.
        order = numpy.argsort(T[:, -1])[::-1]
        T = T[order]
        labels = [labels[i] for i in order]

    for t, label in zip(T, labels):
        plotfun(x, t, label=label)
    if xlabel:
        plt.xlabel(xlabel)
    plt.ylabel('Time in seconds')
    plt.legend()
    return
