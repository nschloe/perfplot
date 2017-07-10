# -*- coding: utf-8 -*-
#
import matplotlib.pyplot as plt
import numpy
import timeit
from tqdm import tqdm


def show(*args, **kwargs):
    _plot(*args, **kwargs)
    plt.show()
    return


def _plot(
        setup, kernels, n_range,
        labels=None,
        xlabel=None,
        repeat=100,
        logx=False,
        logy=False,
        automatic_order=True,
        equality_check=numpy.allclose
        ):
    # Estimate the timer granularity by measuring a no-op.
    noop_time = timeit.repeat(stmt=lambda: None, repeat=10, number=100)
    granularity = max(noop_time) / 100

    timings = numpy.empty((len(kernels), len(n_range), repeat))
    for i, n in enumerate(tqdm(n_range)):
        out = setup(n)
        if equality_check:
            reference = kernels[0](out)
        for k, kernel in enumerate(tqdm(kernels)):
            if equality_check:
                assert equality_check(reference, kernel(out))
            # Make sure that the statement is executed at least so often that
            # the timing exceeds 1000 times the granularity of the clock.
            number = 1
            required_timing = 1000 * granularity
            min_timing = 0.0
            while min_timing <= required_timing:
                timings[k, i] = timeit.repeat(
                    stmt=lambda: kernel(out),
                    repeat=repeat,
                    number=number
                    )
                min_timing = min(timings[k, i])
                timings[k, i] /= number
                # Adapt the number of runs for the next iteration. It needs to
                # be such that the required_timing is just exceeded. If the
                # required timing and minimal timing are just equal, `number`
                # remains the same (up to an allowance of 0.2).
                factor = min(100, required_timing / min_timing + 0.2)
                number = int(factor * number) + 1

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

    if labels is None:
        labels = [k.__name__ for k in kernels]

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
    plt.grid(True)
    plt.legend()
    return
