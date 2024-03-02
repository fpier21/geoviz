from setuptools import setup, find_packages
from os.path import dirname, join, realpath
from textwrap import dedent

PROJECT_ROOT = dirname(realpath(__file__))
REQUIREMENTS_FILE = join(PROJECT_ROOT, "requirements.txt")

with open(REQUIREMENTS_FILE) as f:
    install_reqs = f.read().splitlines()

install_reqs.append("setuptools")

setup(
    name='geoviz',
    version='1.0.0',
    author='Francesco Pierpaoli',
    author_email='francescopierpaoli96@gmail.com',
    url="https://github.com/fpier21/geoviz",
    description='A simple python library for the geometric visualization of the transformations inside a feedforward neural network',
    packages=find_packages(),
    install_requires=install_reqs,
    long_description=dedent(
        """\
       A simple python library for the geometric visualization
       of the transformations inside a feedforward neural network

        Contact
        =============
        If you have any questions or something to report, feel free to write me at
        email: francescopierpaoli96@gmail.com

        This project is hosted at https://github.com/fpier21/geoviz

        The documentation can be found at
        --------"""
    ),
)
