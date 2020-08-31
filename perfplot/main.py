import sys
import time
import timeit

import dufte
import matplotlib
import matplotlib.pyplot as plt
import numpy
import termtables as tt
from tqdm import tqdm

matplotlib.style.use(dufte.style)

# Orders of Magnitude for SI time units in {unit: magnitude} format
si_time = {
    "s": 1e0,  # second
    "ms": 1e-3,  # milisecond
    "us": 1e-6,  # microsecond
    "ns": 1e-9,  # nanosecond
}
if sys.version_info < (3, 7):
    # Make sure that Dictionary is ordered
    from collections import OrderedDict as odict

    si_time = odict(sorted(si_time.items(), key=lambda i: i[1], reverse=True))


def _auto_time_unit(min_time_ns):
    """Automatically obtains a readable unit at which to plot :py:attr:`timings` of the
    benchmarking process. This is accomplished by converting the minimum measured
    execution time into SI second and iterating over the plausible SI time units (s, ms,
    us, ns) to find the first one whos magnitude is smaller than the minimum execution
    time.

    :rtype: str

    .. note::
        This is the same algorithm used by the timeit module
    """
    # Converting minimum timing into seconds from nanoseconds
    t_s = min_time_ns * si_time["ns"]
    for time_unit, magnitude in si_time.items():
        if t_s >= magnitude:
            break
    return time_unit


class PerfplotData:
    def __init__(
        self,
        n_range,
        timings,
        flop,
        labels,
        colors,
        xlabel,
        title,
    ):
        self.n_range = n_range
        self.timings = timings
        self.flop = flop
        self.labels = labels

        self.colors = colors
        if self.colors is None:
            colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
            # extend colors list to fit all self.labels
            colors *= -(-len(self.labels) // len(colors))
            self.colors = colors[: len(self.labels)]

        self.xlabel = xlabel
        self.title = title

    def plot(  # noqa: C901
        self,
        time_unit="s",
        relative_to=None,
        logx="auto",
        logy="auto",
    ):
        if logx == "auto":
            # Check if the x values are approximately equally spaced in log
            log_n_range = numpy.log(self.n_range)
            diff = log_n_range - numpy.linspace(
                log_n_range[0], log_n_range[-1], len(log_n_range)
            )
            logx = numpy.all(numpy.abs(diff) < 1.0e-5)

        if logy == "auto":
            if relative_to is not None:
                logy = False
            elif self.flop is not None:
                logy = False
            else:
                logy = logx

        if logx and logy:
            plotfun = plt.loglog
        elif logx:
            plotfun = plt.semilogx
        elif logy:
            plotfun = plt.semilogy
        else:
            plotfun = plt.plot

        if self.flop is None:
            if relative_to is None:
                # Set time unit of plots. Allowed values: ("s", "ms", "us", "ns", "auto")
                if time_unit == "auto":
                    time_unit = _auto_time_unit(numpy.min(self.timings))
                else:
                    assert time_unit in si_time, "Provided `time_unit` is not valid"

                scaled_timings = self.timings * (si_time["ns"] / si_time[time_unit])
                plt.ylabel(f"Runtime [{time_unit}]")
            else:
                scaled_timings = self.timings / self.timings[relative_to]
                plt.ylabel(f"Runtime relative to {self.labels[relative_to]}()")

            for t, label in zip(scaled_timings, self.labels):
                plotfun(self.n_range, t, label=label)
        else:
            if relative_to is None:
                flops = self.flop / self.timings / si_time["ns"]
                plt.ylabel("FLOPS")
            else:
                flops = self.timings[relative_to] / self.timings
                plt.ylabel(f"FLOPS relative to {self.labels[relative_to]}")

            for fl, label in zip(flops, self.labels):
                plotfun(self.n_range, fl, label=label)

        if self.xlabel:
            plt.xlabel(self.xlabel)
        if self.title:
            plt.title(self.title)
        if relative_to is not None and not logy:
            plt.gca().set_ylim(bottom=0)

        dufte.legend()

    def show(self, **kwargs):
        self.plot(**kwargs)
        plt.show()

    def save(self, filename, transparent=True, bbox_inches="tight", **kwargs):
        self.plot(**kwargs)
        plt.savefig(filename, transparent=transparent, bbox_inches=bbox_inches)
        plt.close()

    def __repr__(self):
        data = numpy.column_stack([self.n_range, self.timings.T])
        return tt.to_string(data, header=["n"] + self.labels, style=None, alignment="r")


def bench(
    setup,
    kernels,
    n_range,
    flops=None,
    labels=None,
    colors=None,
    xlabel=None,
    title=None,
    target_time_per_measurement=1.0,
    equality_check=numpy.allclose,
    show_progress=True,
):
    if labels is None:
        labels = [k.__name__ for k in kernels]

    if hasattr(time, "perf_counter_ns"):
        # New in version 3.7:
        timer = time.perf_counter_ns
        is_ns_timer = True
        resolution = 1  # ns
    else:
        # Remove once we only support 3.7+
        timer = time.perf_counter
        is_ns_timer = False
        # Estimate the timer resolution by measuring a no-op.
        number = 100
        noop_time = timeit.repeat(repeat=10, number=number, timer=timer)
        resolution = numpy.min(noop_time) / number / si_time["ns"]
        # round up to nearest integer
        resolution = -int(-resolution // 1)  # typically around 10 (ns)

    timings = numpy.empty((len(kernels), len(n_range)), dtype=numpy.uint64)

    flop = None if flops is None else numpy.array([flops(n) for n in n_range])

    progress = tqdm if show_progress else lambda x, leave=None: x

    try:
        for i, n in enumerate(progress(n_range)):
            data = setup(n)
            if equality_check:
                relative_to = kernels[0](data)
            for k, kernel in enumerate(
                progress(kernels, leave=(i == len(n_range) - 1))
            ):
                if equality_check:
                    assert kernel(data) is not None, "{} returned None".format(
                        labels[k]
                    )
                    assert equality_check(
                        relative_to, kernel(data)
                    ), "Equality check failure. ({}, {})".format(labels[0], labels[k])

                # First try with one repetition only. If this doesn't exceed the target
                # time, append as many repeats as the first measurements suggests.
                # If the kernel is fast, the measurement with one repetition only can
                # be somewhat off, but most of the time it's good enough.
                remaining_time = int(target_time_per_measurement / si_time["ns"])

                repeat = 1
                t, total_time = _b(data, kernel, repeat, timer, is_ns_timer, resolution)
                time_per_repetition = total_time / repeat

                remaining_time -= total_time
                repeat = int(remaining_time // time_per_repetition)
                if repeat > 0:
                    t2, _ = _b(data, kernel, repeat, timer, is_ns_timer, resolution)
                    t = min(t, t2)

                timings[k, i] = t

    except KeyboardInterrupt:
        timings = timings[:, :i]
        n_range = n_range[:i]

    data = PerfplotData(n_range, timings, flop, labels, colors, xlabel, title)
    return data


def _b(data, kernel, repeat, timer, is_ns_timer, resolution):
    # Make sure that the statement is executed at least so often that the
    # timing exceeds 10 times the resolution of the clock. `number` is larger
    # than 1 only for the fastest computations. Hardly ever happens.
    number = 1
    required_timing = 10 * resolution
    min_timing = 0

    while min_timing <= required_timing:
        tm = numpy.array(
            timeit.repeat(
                stmt=lambda: kernel(data), repeat=repeat, number=number, timer=timer
            )
        )
        if not is_ns_timer:
            tm /= si_time["ns"]
            tm = tm.astype(int)
        min_timing = numpy.min(tm)
        # plt.title("number={} repeat={}".format(number, repeat))
        # plt.semilogy(tm)
        # # plt.hist(tm)
        # plt.show()
        tm //= number
        # Adapt the number of runs for the next iteration such that the
        # required_timing is just exceeded. If the required timing and
        # minimal timing are just equal, `number` remains the same (up
        # to an allowance of 0.2).
        allowance = 0.2
        max_factor = 100
        factor = max_factor
        if min_timing > 0:
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
def plot(
    *args,
    time_unit="s",
    logx="auto",
    logy="auto",
    relative_to=None,
    **kwargs,
):
    out = bench(*args, **kwargs)
    out.plot(
        time_unit=time_unit,
        logx=logx,
        logy=logy,
        relative_to=relative_to,
    )


def show(
    *args,
    time_unit="s",
    relative_to=None,
    logx="auto",
    logy="auto",
    **kwargs,
):
    out = bench(*args, **kwargs)
    out.show(
        time_unit=time_unit,
        relative_to=relative_to,
        logx=logx,
        logy=logy,
    )


def save(
    filename,
    transparent=True,
    *args,
    time_unit="s",
    logx="auto",
    logy="auto",
    relative_to=None,
    **kwargs,
):
    out = bench(*args, **kwargs)
    out.save(
        filename,
        transparent,
        time_unit=time_unit,
        logx=logx,
        logy=logy,
        relative_to=relative_to,
    )
