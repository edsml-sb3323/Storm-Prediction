#!/usr/bin/env python

from setuptools import setup

setup(
    name="Storm Prediction",
    version="1.0",
    description="Storm Evolution Prediction Tool",
    author="ACDS Team Ciaran",
    packages=["Tools/"],
    install_requires=[
        "matplotlib",
        "numpy",
        "pandas",
        "seaborn",
        "pycm",
        "torch",
        "torchvision",
        "livelossplot",
        "itertools",
        "pycm",
        "torchsummary",
        "Pillow",
        "scikit-learn",
        "tqdm",
        "scikit-learn",
    ],
)
