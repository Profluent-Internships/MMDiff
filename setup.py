#!/usr/bin/env python

from setuptools import find_packages, setup

setup(
    name="MMDiff",
    version="0.0.1",
    description="Official PyTorch implementation of 'Joint Sequence-Structure Generation of Nucleic Acid and Protein Complexes with SE(3)-Discrete Diffusion'.",
    author="Alex Morehead",
    author_email="alex.morehead@gmail.com",
    url="https://github.com/Profluent-Internships/MMDiff",
    install_requires=["lightning", "hydra-core"],
    packages=find_packages(),
    # use this to customize global commands available in the terminal after installing the package
    entry_points={
        "console_scripts": [
            "train_command = src.train:main",
            "sample_command = src.sample:main",
        ]
    },
)
