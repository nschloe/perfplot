<p align="center">
  <a href="https://github.com/nschloe/perfplot"><img alt="perfplot" src="https://nschloe.github.io/perfplot/logo-perfplot.svg" width="60%"></a>
</p>

[![PyPi Version](https://img.shields.io/pypi/v/perfplot.svg?style=flat-square)](https://pypi.org/project/perfplot)
[![PyPI pyversions](https://img.shields.io/pypi/pyversions/perfplot.svg?style=flat-square)](https://pypi.org/pypi/perfplot/)
[![GitHub stars](https://img.shields.io/github/stars/nschloe/perfplot.svg?style=flat-square&logo=github&label=Stars&logoColor=white)](https://github.com/nschloe/perfplot)
[![PyPi downloads](https://img.shields.io/pypi/dm/perfplot.svg?style=flat-square)](https://pypistats.org/packages/perfplot)

[![Discord](https://img.shields.io/static/v1?logo=discord&label=chat&message=on%20discord&color=7289da&style=flat-square)](https://discord.gg/hnTJ5MRX2Y)

[![gh-actions](https://img.shields.io/github/workflow/status/nschloe/perfplot/ci?style=flat-square)](https://github.com/nschloe/perfplot/actions?query=workflow%3Aci)
[![codecov](https://img.shields.io/codecov/c/github/nschloe/perfplot.svg?style=flat-square)](https://codecov.io/gh/nschloe/perfplot)
[![LGTM](https://img.shields.io/lgtm/grade/python/github/nschloe/perfplot.svg?style=flat-square)](https://lgtm.com/projects/g/nschloe/perfplot)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg?style=flat-square)](https://github.com/psf/black)

perfplot extends Python's [timeit](https://docs.python.org/3/library/timeit.html) by
testing snippets with input parameters (e.g., the size of an array) and plotting the
results.

For example, to compare different NumPy array concatenation methods, the script
```python
import numpy as np
import perfplot

perfplot.show(
    setup=lambda n: np.random.rand(n),  # or setup=np.random.rand
    kernels=[
        lambda a: np.c_[a, a],
        lambda a: np.stack([a, a]).T,
        lambda a: np.vstack([a, a]).T,
        lambda a: np.column_stack([a, a]),
        lambda a: np.concatenate([a[:, None], a[:, None]], axis=1),
    ],
    labels=["c_", "stack", "vstack", "column_stack", "concat"],
    n_range=[2 ** k for k in range(25)],
    xlabel="len(a)",
    # More optional arguments with their default values:
    # logx="auto",  # set to True or False to force scaling
    # logy="auto",
    # equality_check=np.allclose,  # set to None to disable "correctness" assertion
    # show_progress=True,
    # target_time_per_measurement=1.0,
    # max_time=None,  # maximum time per measurement
    # time_unit="s",  # set to one of ("auto", "s", "ms", "us", or "ns") to force plot units
    # relative_to=1,  # plot the timings relative to one of the measurements
    # flops=lambda n: 3*n,  # FLOPS plots
)
```
produces

![](https://nschloe.github.io/perfplot/concat.svg) | ![](https://nschloe.github.io/perfplot/relative.svg)
| --- | --- |

Clearly, `stack` and `vstack` are the best options for large arrays.

(By default, perfplot asserts the equality of the output of all snippets, too.)

If your plot takes a while to generate, you can also use
<!--exdown-skip-->
```python
perfplot.live(
    # ...
)
```
<img alt="live" src="https://nschloe.github.io/perfplot/live.gif" width="40%">

with the same arguments as above. It will plot the updates live.

Benchmarking and plotting can be separated. This allows multiple plots of the same data,
for example:
<!--exdown-skip-->
```python
out = perfplot.bench(
    # same arguments as above (except the plot-related ones, like time_unit or log*)
)
out.show()
out.save("perf.png", transparent=True, bbox_inches="tight")
```

Other examples:

  * [Making a flat list out of list of lists in Python](https://stackoverflow.com/a/45323085/353337)
  * [Most efficient way to map function over numpy array](https://stackoverflow.com/a/46470401/353337)
  * [numpy: most efficient frequency counts for unique values in an array](https://stackoverflow.com/a/43096495/353337)
  * [Most efficient way to reverse a numpy array](https://stackoverflow.com/a/44921013/353337)
  * [How to add an extra column to an numpy array](https://stackoverflow.com/a/40218298/353337)
  * [Initializing numpy matrix to something other than zero or one](https://stackoverflow.com/a/45006691/353337)

### Installation

perfplot is [available from the Python Package
Index](https://pypi.org/project/perfplot/), so simply do
```
pip install perfplot
```
to install.

### Testing

To run the perfplot unit tests, check out this repository and type
```
pytest
```

### License
This software is published under the [GPLv3 license](https://www.gnu.org/licenses/gpl-3.0.en.html).
