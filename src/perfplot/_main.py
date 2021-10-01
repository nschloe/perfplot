from __future__ import annotations

import io
import time
import timeit
from typing import Callable

try:
    # Python 3.8+
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

import dufte
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from rich.console import Console
from rich.progress import Progress
from rich.table import Table

from ._exceptions import PerfplotError

plt.style.use(dufte.style)

# Orders of Magnitude for SI time units in {unit: magnitude} format
si_time = {
    "s": 1e0,  # second
    "ms": 1e-3,  # milisecond
    "us": 1e-6,  # microsecond
    "ns": 1e-9,  # nanosecond
}


def _auto_time_unit(time_s: float) -> str:
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
    time_unit = None
    for time_unit, magnitude in si_time.items():
        if time_s >= magnitude:
            break
    assert time_unit is not None
    return time_unit


class PerfplotData:
    def __init__(
        self,
        n_range: list[int],
        timings_s,
        flop,
        labels: list[str],
        xlabel: str | None,
    ):
        self.n_range = np.asarray(n_range)
        self.timings_s = timings_s
        self.flop = flop
        self.labels = labels
        self.xlabel = xlabel

    def plot(  # noqa: C901
        self,
        time_unit: str = "s",
        relative_to: int | None = None,
        logx: str | bool = "auto",
        logy: str | bool = "auto",
    ):
        if logx == "auto":
            # Check if the x values are approximately equally spaced in log
            if np.any(self.n_range <= 0):
                logx = False
            else:
                log_n_range = np.log(self.n_range)
                linlog = np.linspace(log_n_range[0], log_n_range[-1], len(log_n_range))
                # don't consider first and last, they are equal anyway
                rel_diff = (log_n_range - linlog)[1:-1] / log_n_range[1:-1]
                logx = np.all(np.abs(rel_diff) <= 0.1)

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
                # Set time unit of plots.
                # Allowed values: ("s", "ms", "us", "ns", "auto")
                if time_unit == "auto":
                    time_unit = _auto_time_unit(np.min(self.timings_s))
                else:
                    assert time_unit in si_time, "Provided `time_unit` is not valid"

                scaled_timings = self.timings_s / si_time[time_unit]
                ylabel = f"Runtime [{time_unit}]"
            else:
                scaled_timings = self.timings_s / self.timings_s[relative_to]
                ylabel = f"Runtime\nrelative to {self.labels[relative_to]}"

            for t, label in zip(scaled_timings, self.labels):
                plotfun(self.n_range, t, label=label)

            dufte.ylabel(ylabel)
        else:
            if relative_to is None:
                flops = self.flop / self.timings_s
                plt.title("FLOPS")
            else:
                flops = self.timings_s[relative_to] / self.timings_s
                plt.title(f"FLOPS relative to {self.labels[relative_to]}")

            for fl, label in zip(flops, self.labels):
                plotfun(self.n_range, fl, label=label)

        if self.xlabel:
            plt.xlabel(self.xlabel)
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
        table = Table(show_header=True)
        table.add_column("n")
        for label in self.labels:
            table.add_column(label)

        for n, t in zip(self.n_range, self.timings_s.T):
            lst = [str(n)] + [str(tt) for tt in t]
            table.add_row(*lst)

        f = io.StringIO()
        console = Console(file=f)
        console.print(table)
        return f.getvalue()


# iterator
class Bench:
    def __init__(
        self,
        setup,
        kernels,
        equality_check,
        n_range,
        target_time_per_measurement,
        max_time,
        labels,
        cutoff_reached,
        callback=None,
    ):
        self.setup = setup
        self.kernels = kernels
        self.equality_check = equality_check
        # Make n_range a list of "scalar" np.arrays. This makes it possible for the
        # setup function to override the value with
        # n[...] = something_else.
        # That is useful if you want to plot the graphs against something else than the
        # input n.
        assert len(np.asarray(n_range).shape) == 1
        self.n_range = [np.array(item) for item in n_range]
        self.target_time_per_measurement = target_time_per_measurement
        self.max_time = max_time
        self.labels = labels
        self.cutoff_reached = cutoff_reached
        self.callback = callback

        self.idx = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.idx >= len(self.n_range):
            raise StopIteration

        # Explicitly convert to Python int. One difference between a numpy.array(int)
        # and a native Python int are in expressions like `n * [2.0]`.
        n = int(self.n_range[self.idx])

        self.idx += 1

        data = None
        if callable(self.setup):
            data = self.setup(n)

        reference = None
        timings = []
        for k, kernel in enumerate(self.kernels):
            if self.cutoff_reached[k]:
                timings.append(np.nan)
                if self.callback is not None:
                    self.callback()
                continue

            if isinstance(self.setup, list):
                data = self.setup[k](n)

            # First let the function run once. The value is used for the equality_check
            # and the time for gauging how many more repetitions are to be done. If the
            # initial time doesn't exceed the target time, append as many repetitions as
            # the first measurement suggests. If the kernel is fast, the measurement
            # with one repetition only can be somewhat off because the CPU needs to spin
            # up first. The actual times are only reached after a few hundred
            # nanoseconds of computation. Most of the time it's okay though.
            t0_ns = time.perf_counter_ns()
            val = kernel(data)
            t1_ns = time.perf_counter_ns()
            t_ns = t1_ns - t0_ns

            if t_ns == 0:
                raise RuntimeError("Measured 0 ns for a function call. Try again?")

            if self.equality_check:
                if k == 0:
                    reference = val
                else:
                    try:
                        is_equal = self.equality_check(reference, val)
                    except TypeError:
                        raise PerfplotError(
                            "Error in equality_check. "
                            + "Try setting equality_check=None."
                        )
                    else:
                        if not is_equal:
                            raise PerfplotError(
                                "Equality check failure. "
                                + f"({self.labels[0]}, {self.labels[k]})"
                            )

            # First try with one repetition only.
            remaining_time_ns = int(self.target_time_per_measurement / si_time["ns"])

            if self.max_time is not None and t_ns * si_time["ns"] > self.max_time:
                self.cutoff_reached[k] = True

            remaining_time_ns -= t_ns
            repeat = remaining_time_ns // t_ns
            if repeat > 0:
                t2 = _b(data, kernel, repeat)
                t_ns = min(t_ns, t2)

            timings.append(t_ns)
            if self.callback is not None:
                self.callback()

        return np.array(timings) * 1.0e-9


def _b(data, kernel: Callable, repeat: int):
    # Make sure that the statement is executed at least so often that the timing exceeds
    # 10 times the resolution of the clock. `number` is larger than 1 only for the
    # fastest computations. Hardly ever happens.
    number = 1
    required_timing_ns = 10
    min_timing_ns = 0
    tm = None

    while min_timing_ns <= required_timing_ns:
        tm = np.array(
            timeit.repeat(
                stmt=lambda: kernel(data),
                repeat=repeat,
                number=number,
                timer=time.perf_counter_ns,
            )
        )
        min_timing_ns = np.min(tm)
        tm //= number
        # Adapt the number of runs for the next iteration such that the
        # required_timing_ns is just exceeded. If the required timing and minimal timing
        # are just equal, `number` remains the same (up to an allowance of 0.2).
        allowance = 0.2
        max_factor = 100
        factor = max_factor
        if min_timing_ns > 0:
            # The next expression is
            #   min(max_factor, required_timing_ns / min_timing_ns + allowance)
            # with avoiding division by 0 if min_timing_ns is too small.
            factor = (
                required_timing_ns / min_timing_ns + allowance
                if min_timing_ns > required_timing_ns / (max_factor - allowance)
                else max_factor
            )

        number = int(factor * number) + 1

    assert tm is not None
    # Only return the minimum time; everthing else just measures how slow the system can
    # go.
    return np.min(tm)


def live(
    setup: Callable,
    kernels: list[Callable],
    n_range: npt.ArrayLike,
    labels: list[str] | None = None,
    xlabel: str | None = None,
    target_time_per_measurement: float = 1.0,
    max_time: float | None = None,
    equality_check: Callable | None = np.allclose,
    show_progress: bool = True,
    logx: Literal["auto"] | bool = "auto",
    logy: Literal["auto"] | bool = "auto",
):
    if labels is None:
        labels = [k.__name__ for k in kernels]

    # for data type of n_range, see above
    assert len(np.asarray(n_range).shape) == 1
    n_range = [np.array(item) for item in n_range]

    if logx == "auto":
        # Check if the x values are approximately equally spaced in log
        if any(n <= 0 for n in n_range):
            logx = False
        else:
            log_n_range = np.log(n_range)
            diff = log_n_range - np.linspace(
                log_n_range[0], log_n_range[-1], len(log_n_range)
            )
            logx = np.all(np.abs(diff) < 1.0e-5)

    if logy == "auto":
        logy = logx

    # animation adapted from
    # <https://matplotlib.org/stable/gallery/animation/animate_decay.html>
    with Progress() as progress:
        if show_progress:
            task1 = progress.add_task("Overall", total=len(n_range))
            task2 = progress.add_task("Kernels", total=len(kernels))

        def init():
            for line, yd in zip(lines, ydata):
                line.set_data(xdata, yd)
            return lines

        fig, ax = plt.subplots()

        if logx and logy:
            plotfun = ax.loglog
        elif logx:
            plotfun = ax.semilogx
        elif logy:
            plotfun = ax.semilogy
        else:
            plotfun = ax.plot

        lines = []
        for label in labels:
            lines.append(plotfun([], [], label=label)[0])

        ax.legend()
        dufte.ylabel("Runtime [s]")
        if xlabel:
            ax.set_xlabel(xlabel)
        xdata = []
        ydata = []
        for _ in range(len(kernels)):
            ydata.append([])

        def run(data):
            # update the data
            for t, yd in zip(data, ydata):
                yd.append(t)

            xdata = n_range[: len(ydata[0])]

            if len(xdata) > 1:
                ax.set_xlim(np.nanmin(xdata), np.nanmax(xdata))

            yd = np.asarray(ydata)
            if yd.size > 1:
                ax.set_ylim(np.nanmin(yd), np.nanmax(yd))

            ax.figure.canvas.draw()

            for line, yd in zip(lines, ydata):
                line.set_data(xdata, yd)

            if show_progress:
                progress.update(task1, advance=1)
                progress.reset(task2)
            return lines

        def callback():
            if show_progress:
                progress.update(task2, advance=1)

        bench = Bench(
            setup,
            kernels,
            equality_check,
            n_range,
            target_time_per_measurement,
            max_time,
            labels,
            cutoff_reached=np.zeros(len(n_range), dtype=bool),
            callback=callback,
        )
        # Assign FuncAnimation to a dummy variable to avoid it being destroyed before
        # the animation has completed. This is mpl's recommendation.
        _ = animation.FuncAnimation(
            fig, run, bench, interval=10, init_func=init, repeat=False
        )

        # anim.save("anim.gif", fps=5)
        plt.show()


def bench(
    kernels: list[Callable],
    n_range: list[int],
    setup: Callable | list[Callable] | None = None,
    flops: Callable | None = None,
    labels: list[str] | None = None,
    xlabel: str | None = None,
    target_time_per_measurement: float = 1.0,
    max_time: float | None = None,
    equality_check: Callable | None = np.allclose,
    show_progress: bool = True,
):
    if labels is None:
        labels = [k.__name__ for k in kernels]

    timings_s = np.full((len(n_range), len(kernels)), np.nan)
    cutoff_reached = np.zeros(len(kernels), dtype=bool)

    flop = None if flops is None else np.array([flops(n) for n in n_range])

    task1 = None
    task2 = None

    # inner iterator
    with Progress() as progress:
        try:
            if show_progress:
                task1 = progress.add_task("Overall", total=len(n_range))
                task2 = progress.add_task("Kernels", total=len(kernels))

            def callback():
                if show_progress:
                    assert task2 is not None
                    progress.update(task2, advance=1)

            b = Bench(
                setup,
                kernels,
                equality_check,
                n_range,
                target_time_per_measurement,
                max_time,
                labels,
                cutoff_reached,
                callback=callback,
            )

            for i in range(len(n_range)):
                timings_s[i] = next(b)
                # override n_rane in case it got overridden in next()
                n_range = b.n_range

                if show_progress:
                    assert task1 is not None
                    assert task2 is not None
                    progress.update(task1, advance=1)
                    progress.reset(task2)

        except KeyboardInterrupt:
            timings_s = timings_s[:i]
            n_range = n_range[:i]

    return PerfplotData(n_range, timings_s.T, flop, labels, xlabel)


# For backward compatibility:
def plot(
    *args,
    time_unit: str = "s",
    logx: Literal["auto"] | bool = "auto",
    logy: Literal["auto"] | bool = "auto",
    relative_to: int | None = None,
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
    time_unit: str = "s",
    relative_to: int | None = None,
    logx: bool | Literal["auto"] = "auto",
    logy: bool | Literal["auto"] = "auto",
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
    time_unit: str = "s",
    logx: bool | Literal["auto"] = "auto",
    logy: bool | Literal["auto"] = "auto",
    relative_to: int | None = None,
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
