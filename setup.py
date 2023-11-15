# !/usr/bin/python3
import os
from setuptools import setup, find_packages

with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name="irtoolkit",
    version="0.1",
    description="Toolkit for chemical imaging",
    url="https://github.com/stefangstark/mouse-placenta-development",
    author="Stefan G. Stark",
    author_email="starks@ethz.ch",
    license="BSD",
    packages=find_packages(),
    test_suite="nose.collector",
    tests_require=["nose"],
    install_requires=required,
    zip_safe=False,
)
