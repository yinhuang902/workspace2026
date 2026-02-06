from setuptools import setup, find_packages
setup(
    name="snoglode",
    author="georgia stinchfield",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pyomo>=6.6",
        "numpy",
        "pandas",
        "matplotlib"
    ],
)