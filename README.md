# perfplot

[![Build Status](https://travis-ci.org/nschloe/perfplot.svg?branch=master)](https://travis-ci.org/nschloe/perfplot)
[![codecov](https://img.shields.io/codecov/c/github/nschloe/perfplot.svg)](https://codecov.io/gh/nschloe/perfplot)
[![PyPi Version](https://img.shields.io/pypi/v/perfplot.svg)](https://pypi.python.org/pypi/perfplot)
[![GitHub stars](https://img.shields.io/github/stars/nschloe/perfplot.svg?style=social&label=Stars)](https://github.com/nschloe/perfplot)

perfplot extends Python's very own
[timeit](https://docs.python.org/2/library/timeit.html) by testing snippets
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

Clearly, `stack` and `vstack` are the best options for large arrays!

### Installation

#### Python Package Index

perfplot is [available from the Python Package
Index](https://pypi.python.org/pypi/perfplot/), so simply do
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
