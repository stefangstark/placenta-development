# !/usr/bin/python3

from setuptools import setup, find_packages


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
    install_requires=[],
    zip_safe=False,
)
