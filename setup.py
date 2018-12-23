#!/user/bin/env python

from setuptools import find_packages, setup

project = "word2vec-pytorch"
version = "0.1.0"

setup(
    name=project,
    version=version,
    packages=find_packages(exclude=["*.tests", "*.tests.*", "tests.*", "tests"]),
    include_package_data=True,
    zip_safe=False,
    install_requires=[
        "numpy>=1.14.3",
        "torch>=1.0.0",
        "tqdm>=4.28.0",
        "Click>=7.0",
    ],
    setup_requires=[
        "nose>=1.3.7",
    ],
    tests_require=[
        "coverage>=3.7.1",
        "PyHamcrest>=1.9.0",
    ],

    entry_points="""
        [console_scripts]
        word2vec=word2vec.main:word2vec
    """,
)