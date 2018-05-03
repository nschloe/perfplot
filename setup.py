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
    return codecs.open(os.path.join(base_dir, fname), encoding='utf-8').read()


setup(
    name='perfplot',
    version=about['__version__'],
    author=about['__author__'],
    author_email=about['__author_email__'],
    packages=find_packages(),
    description='Performance plots for Python code snippets',
    long_description=read('README.md'),
    long_description_content_type='text/markdown',
    url='https://github.com/nschloe/perfplot',
    license=about['__license__'],
    platforms='any',
    install_requires=[
        'matplotlib',
        'numpy',
        'pipdate >=0.2.0, <0.3.0',
        'tqdm',
        ],
    extras_require={
        'all': ['pandas'],
        'print': ['pandas'],
        },
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
