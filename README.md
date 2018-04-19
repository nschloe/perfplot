# perfplot

[![CircleCI](https://img.shields.io/circleci/project/github/nschloe/perfplot/master.svg)](https://circleci.com/gh/nschloe/perfplot/tree/master)
[![codecov](https://img.shields.io/codecov/c/github/nschloe/perfplot.svg)](https://codecov.io/gh/nschloe/perfplot)
[![Codacy grade](https://img.shields.io/codacy/grade/32994ce499db42059777d42edcfce900.svg)](https://app.codacy.com/app/nschloe/perfplot/dashboard)
[![PyPi Version](https://img.shields.io/pypi/v/perfplot.svg)](https://pypi.org/project/perfplot)
[![GitHub stars](https://img.shields.io/github/stars/nschloe/perfplot.svg?logo=github&label=Stars)](https://github.com/nschloe/perfplot)

perfplot extends Python's
[timeit](https://docs.python.org/3/library/timeit.html) by testing snippets
with input parameters (e.g., the size of an array) and plotting the results.
(By default, perfplot asserts the equality of the output of all snippets, too.)

For example, to compare different NumPy array concatenation methods, the script
```python
import numpy
import perfplot

perfplot.show(
    setup=numpy.random.rand,
    kernels=[
        lambda a: numpy.c_[a, a],
        lambda a: numpy.stack([a, a]).T,
        lambda a: numpy.vstack([a, a]).T,
        lambda a: numpy.column_stack([a, a]),
        lambda a: numpy.concatenate([a[:, None], a[:, None]], axis=1)
        ],
    labels=['c_', 'stack', 'vstack', 'column_stack', 'concat'],
    n_range=[2**k for k in range(15)],
    xlabel='len(a)'
    )
```
produces

![](https://nschloe.github.io/perfplot/concat.png)

Clearly, `stack` and `vstack` are the best options for large arrays.

Benchmarking and plotting can be separated, too. This allows multiple plots of
the same data, for example:
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
pip install -U perfplot
```
to install or upgrade.

### Testing

To run the perfplot unit tests, check out this repository and type
```
pytest
```

### Distribution
To create a new release

1. bump the `__version__` number,

2. publish to PyPi and tag on GitHub:
    ```
    $ make publish
    ```

### License

perfplot is published under the [MIT license](https://en.wikipedia.org/wiki/MIT_License).
