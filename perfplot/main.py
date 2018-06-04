# -*- coding: utf-8 -*-
#
import timeit

import matplotlib.pyplot as plt
import numpy
from tqdm import tqdm


class PerfplotData(object):
    def __init__(
        self,
        n_range,
        timings,
        labels,
        colors,
        xlabel,
        title,
        logx,
        logy,
        automatic_order,
    ):
        self.n_range = n_range
        self.timings = timings
        self.labels = labels

        self.colors = colors
        if self.colors is None:
            prop_cycle = plt.rcParams["axes.prop_cycle"]
            self.colors = prop_cycle.by_key()["color"][: len(self.labels)]

        self.xlabel = xlabel
        self.title = title

        # choose plot function
        if logx and logy:
            self.plotfun = plt.loglog
        elif logx:
            self.plotfun = plt.semilogx
        elif logy:
            self.plotfun = plt.semilogy
        else:
            self.plotfun = plt.plot

        if automatic_order:
            # Sort timings by the last entry. This makes the order in the
            # legend correspond to the order of the lines.
            order = numpy.argsort(self.timings[:, -1])[::-1]
            self.timings = self.timings[order]
            self.labels = [self.labels[i] for i in order]
            self.colors = [self.colors[i] for i in order]
        return

    def plot(self):
        for t, label, color in zip(self.timings, self.labels, self.colors):
            self.plotfun(self.n_range, t, label=label, color=color)
        if self.xlabel:
            plt.xlabel(self.xlabel)
        if self.title:
            plt.title(self.title)
        plt.ylabel("Time in seconds")
        plt.grid(True)
        plt.legend()
        return

    def show(self):
        self.plot()
        plt.show()
        return

    def save(self, filename, transparent=True):
        self.plot()
        plt.savefig(filename, transparent=transparent)
        return

    def __repr__(self):
        import pandas

        return pandas.DataFrame(self.timings.T, self.n_range, self.labels).to_string()


def bench(
    setup,
    kernels,
    n_range,
    labels=None,
    colors=None,
    xlabel=None,
    title=None,
    repeat=None,
    target_time_per_measurement=None,
    logx=False,
    logy=False,
    automatic_order=True,
    equality_check=numpy.allclose,
):
    if repeat is None:
        if target_time_per_measurement is None:
            target_time_per_measurement = 1.0
    else:
        assert (
            target_time_per_measurement is None
        ), "Only one of `repeat` ({}) and `target_time_per_measurement ({}) can be specified.".format(
            repeat, target_time_per_measurement
        )

    if labels is None:
        labels = [k.__name__ for k in kernels]

    # Estimate the timer resolution by measuring a no-op.
    # TODO Python 3.7 will feature a nanosecond timer. Use that when available.
    noop_time = timeit.repeat(repeat=10, number=100)
    resolution = numpy.min(noop_time) / 100

    timings = numpy.empty((len(kernels), len(n_range)))

    last_repeat = numpy.empty(len(kernels), dtype=int)
    last_n = numpy.empty(len(kernels), dtype=int)
    last_total_time = numpy.empty(len(kernels))

    try:
        for i, n in enumerate(tqdm(n_range)):
            data = setup(n)
            if equality_check:
                reference = kernels[0](data)
            for k, kernel in enumerate(tqdm(kernels)):
                if equality_check:
                    assert equality_check(
                        reference, kernel(data)
                    ), "Equality check failure. ({}, {})".format(labels[0], labels[k])

                if repeat is None:
                    if i == 0:
                        # Bootstrap the repeat count and timing
                        last_repeat[k] = 1
                        last_n[k] = n
                        _, last_total_time[k] = _b(
                            data, kernel, last_repeat[k], resolution
                        )
                    # Set the number of repetitions such that it would hit
                    # target_time_per_measurement if the timing scales linearly
                    # with n.
                    rp = (
                        last_repeat[k]
                        * target_time_per_measurement
                        / last_total_time[k]
                        * last_n[k]
                        / n
                    )
                    # Round up
                    rp = -int(-rp // 1)
                else:
                    # Fixed number of repeats
                    rp = repeat

                timings[k, i], last_total_time[k] = _b(data, kernel, rp, resolution)
                last_repeat[k] = rp
                last_n[k] = n

    except KeyboardInterrupt:
        timings = timings[:, :i]
        n_range = n_range[:i]

    data = PerfplotData(
        n_range, timings, labels, colors, xlabel, title, logx, logy, automatic_order
    )
    return data


def _b(data, kernel, repeat, resolution):
    # Make sure that the statement is executed at least so often that the
    # timing exceeds 10 times the resolution of the clock. `number` is larger
    # than 1 only for the fastest computations. Hardly ever happens.
    number = 1
    required_timing = 10 * resolution
    min_timing = 0.0
    while min_timing <= required_timing:
        tm = numpy.array(
            timeit.repeat(stmt=lambda: kernel(data), repeat=repeat, number=number)
        )
        min_timing = numpy.min(tm)
        # plt.title("number={} repeat={}".format(number, repeat))
        # plt.semilogy(tm)
        # # plt.hist(tm)
        # plt.show()
        tm /= number
        # Adapt the number of runs for the next iteration such that the
        # required_timing is just exceeded. If the required timing and
        # minimal timing are just equal, `number` remains the same (up
        # to an allowance of 0.2).
        allowance = 0.2
        max_factor = 100
        # The next expression is
        #   min(max_factor, required_timing / min_timing + allowance)
        # with avoiding division by 0 if min_timing is too small.
        factor = (
            required_timing / min_timing + allowance
            if min_timing > required_timing / (max_factor - allowance)
            else max_factor
        )
        number = int(factor * number) + 1
    # Only return the minimum time; everthing else just measures
    # how slow the system can go.
    return numpy.min(tm), numpy.sum(tm)


# For backward compatibility:
def plot(*args, **kwargs):
    out = bench(*args, **kwargs)
    out.plot()
    return


def show(*args, **kwargs):
    out = bench(*args, **kwargs)
    out.show()
    return


def save(filename, *args, **kwargs):
    out = bench(*args, **kwargs)
    out.save(filename)
    return
