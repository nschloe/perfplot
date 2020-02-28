import os

from setuptools import find_packages, setup

# https://packaging.python.org/single_source_version/
base_dir = os.path.abspath(os.path.dirname(__file__))
about = {}
with open(os.path.join(base_dir, "perfplot", "__about__.py"), "rb") as f:
    exec(f.read(), about)


setup(
    name="perfplot",
    version=about["__version__"],
    author=about["__author__"],
    author_email=about["__author_email__"],
    packages=find_packages(),
    description="Performance plots for Python code snippets",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/nschloe/perfplot",
    license=about["__license__"],
    platforms="any",
    install_requires=["matplotlib", "numpy", "tqdm", "termtables"],
    python_requires=">=3.5",
    classifiers=[
        about["__status__"],
        about["__license__"],
        "Intended Audience :: Developers",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Topic :: Software Development",
        "Topic :: Utilities",
    ],
)
