# -*- coding: utf-8 -*-
#
import timeit

import matplotlib.pyplot as plt
import numpy
from tqdm import tqdm


class PerfplotData(object):
    def __init__(self, n_range, timings,
                 labels,
                 colors,
                 xlabel,
                 title,
                 repeat,
                 logx,
                 logy,
                 automatic_order):
        self.n_range = n_range
        self.timings = timings
        self.labels = labels
        self.colors = colors
        self.xlabel = xlabel
        self.title = title
        self.logx = logx
        self.logy = logy
        self.automatic_order = automatic_order
        return

    def plot(self):
        # choose plot function
        if self.logx and self.logy:
            plotfun = plt.loglog
        elif self.logx:
            plotfun = plt.semilogx
        elif self.logy:
            plotfun = plt.semilogy
        else:
            plotfun = plt.plot
        # plot minimum time
        x = self.n_range
        T = numpy.min(self.timings, axis=2)

        if self.colors is None:
            prop_cycle = plt.rcParams['axes.prop_cycle']
            colors = prop_cycle.by_key()['color'][:len(self.labels)]

        if self.automatic_order:
            # Sort T by the last entry. This makes the order in the legend
            # correspond to the order of the lines.
            order = numpy.argsort(T[:, -1])[::-1]
            T = T[order]
            self.labels = [self.labels[i] for i in order]
            colors = [colors[i] for i in order]

        for t, label, color in zip(T, self.labels, colors):
            plotfun(x, t, label=label, color=color)
        if self.xlabel:
            plt.xlabel(self.xlabel)
        if self.title:
            plt.title(self.title)
        plt.ylabel('Time in seconds')
        plt.grid(True)
        plt.legend()
        return

    def show(self):
        self.plot()
        plt.show()
        return

    def save(self, filename):
        self.plot()
        plt.savefig(filename, transparent=True)
        return


# pylint: disable=too-many-arguments,too-many-locals,too-many-branches
def bench(setup, kernels, n_range,
          labels=None,
          colors=None,
          xlabel=None,
          title=None,
          repeat=100,
          logx=False,
          logy=False,
          automatic_order=True,
          equality_check=numpy.allclose):
    if labels is None:
        labels = [k.__name__ for k in kernels]

    # Estimate the timer granularity by measuring a no-op.
    noop_time = timeit.repeat(repeat=10, number=100)
    granularity = min(noop_time) / 100

    timings = numpy.empty((len(kernels), len(n_range), repeat))
    for i, n in enumerate(tqdm(n_range)):
        out = setup(n)
        if equality_check:
            reference = kernels[0](out)
        for k, kernel in enumerate(tqdm(kernels)):
            if equality_check:
                assert equality_check(reference, kernel(out)), \
                    'Equality check fail. ({}, {})' \
                    .format(labels[0], labels[k])
            # Make sure that the statement is executed at least so often that
            # the timing exceeds 1000 times the granularity of the clock.
            number = 1
            required_timing = 1000 * granularity
            min_timing = 0.0
            while min_timing <= required_timing:
                timings[k, i] = timeit.repeat(
                    # pylint: disable=cell-var-from-loop
                    stmt=lambda: kernel(out),
                    repeat=repeat,
                    number=number
                    )
                min_timing = min(timings[k, i])
                # print(timings[k, i])
                # plt.semilogy(range(len(timings[k, i])), timings[k, i])
                # plt.hist(timings[k, i])
                # plt.show()
                timings[k, i] /= number
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

    data = PerfplotData(
        n_range,
        timings,
        labels,
        colors,
        xlabel,
        title,
        repeat,
        logx,
        logy,
        automatic_order
        )
    return data
