#!/usr/bin/env python

from setuptools import find_packages, setup

setup(
    name="MMDiff",
    version="0.0.1",
    description="SE(3) Diffusion for design of protein-nucleic acid complexes",
    author="",
    author_email="amorehead@profluent.bio",
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
