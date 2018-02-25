# -*- coding: utf-8 -*-
#
import os
import codecs

from setuptools import setup, find_packages

# https://packaging.python.org/single_source_version/
base_dir = os.path.abspath(os.path.dirname(__file__))
about = {}
with open(os.path.join(base_dir, 'perfplot', '__about__.py'), 'rb') as f:
    # pylint: disable=exec-used
    exec(f.read(), about)


def read(fname):
    try:
        content = codecs.open(
            os.path.join(os.path.dirname(__file__), fname),
            encoding='utf-8'
            ).read()
    except IOError:
        content = ''
    return content


setup(
    name='perfplot',
    version=about['__version__'],
    author=about['__author__'],
    author_email=about['__author_email__'],
    packages=find_packages(),
    description='Performance plots for Python code snippets',
    long_description=read('README.rst'),
    url='https://github.com/nschloe/perfplot',
    download_url='https://pypi.python.org/pypi/perfplot',
    license=about['__license__'],
    platforms='any',
    install_requires=[
        'matplotlib',
        'numpy',
        'pipdate',
        'tqdm',
        ],
    classifiers=[
        about['__status__'],
        about['__license__'],
        'Intended Audience :: Developers',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 3',
        'Topic :: Software Development',
        'Topic :: Utilities'
        ]
    )
