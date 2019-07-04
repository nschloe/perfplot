<p align="center">
  <img alt="perfplot" src="https://nschloe.github.io/perfplot/logo-perfplot.svg" width="60%">
</p>

[![CircleCI](https://img.shields.io/circleci/project/github/nschloe/perfplot/master.svg?style=flat-square)](https://circleci.com/gh/nschloe/perfplot/tree/master)
[![codecov](https://img.shields.io/codecov/c/github/nschloe/perfplot.svg?style=flat-square)](https://codecov.io/gh/nschloe/perfplot)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg?style=flat-square)](https://github.com/ambv/black)
[![PyPi Version](https://img.shields.io/pypi/v/perfplot.svg?style=flat-square)](https://pypi.org/project/perfplot)
[![GitHub stars](https://img.shields.io/github/stars/nschloe/perfplot.svg?style=flat-square&logo=github&label=Stars&logoColor=white)](https://github.com/nschloe/perfplot)
[![PyPi downloads](https://img.shields.io/pypi/dd/perfplot.svg?style=flat-square)](https://pypistats.org/packages/perfplot)

perfplot extends Python's [timeit](https://docs.python.org/3/library/timeit.html) by
testing snippets with input parameters (e.g., the size of an array) and plotting the
results.  (By default, perfplot asserts the equality of the output of all snippets,
too.)

For example, to compare different NumPy array concatenation methods, the script
```python
import numpy
import perfplot

perfplot.show(
    setup=lambda n: numpy.random.rand(n),  # or simply setup=numpy.random.rand
    kernels=[
        lambda a: numpy.c_[a, a],
        lambda a: numpy.stack([a, a]).T,
        lambda a: numpy.vstack([a, a]).T,
        lambda a: numpy.column_stack([a, a]),
        lambda a: numpy.concatenate([a[:, None], a[:, None]], axis=1),
    ],
    labels=["c_", "stack", "vstack", "column_stack", "concat"],
    n_range=[2 ** k for k in range(15)],
    xlabel="len(a)",
    # More optional arguments with their default values:
    # title=None,
    # logx=False,
    # logy=False,
    # equality_check=numpy.allclose,  # set to None to disable "correctness" assertion
    # automatic_order=True,
    # colors=None,
    # target_time_per_measurement=1.0,
)
```
produces

![](https://nschloe.github.io/perfplot/concat.png)

Clearly, `stack` and `vstack` are the best options for large arrays.

Benchmarking and plotting can be separated, too. This allows multiple plots of the same
data, for example:
```python
out = perfplot.bench(
    # same arguments as above
    )
out.show()
out.save('perf.png')
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
pip3 install perfplot --user
```
to install or upgrade.

### Testing

To run the perfplot unit tests, check out this repository and type
```
pytest
```

### License

perfplot is published under the [MIT license](https://en.wikipedia.org/wiki/MIT_License).
