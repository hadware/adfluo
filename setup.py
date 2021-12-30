# Always prefer setuptools over distutils
from pathlib import Path

from setuptools import setup, find_packages

here = Path(__file__).parent.absolute()

# Get the long description from the README file
with open(here / Path('README.md')) as f:
    long_description = f.read()

setup(
    name='adfluo',
    version='0.1.0',
    description='Pipeline-oriented feature extraction for multimodal datasets',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/bootphon/adfluo',
    author='Hadrien Titeux',
    author_email='hadrien.titeux@ens.fr',
    license="MIT",
    classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3.6'
        'Programming Language :: Python :: 3.7'
        'Programming Language :: Python :: 3.8'
        'Programming Language :: Python :: 3.9'
    ],
    keywords='',
    packages=find_packages(exclude=['docs', 'tests']),
    setup_requires=['setuptools>=38.6.0'],  # >38.6.0 needed for markdown README.md
    install_requires=[
        "tqdm",
        "dataclasses; python_version <'3.7'",
        "sortedcontainers>=2.3.0",
        "typing-extensions"
    ],
    extras_requires={
        "plot": [
            "matplotlib",
            "networkX"
        ],
        "tests": [
            "pytest",
            "pandas",
            "hdf5"
        ],
        "docs": [
            "sphinx",
            "sphinx_rtd_theme"
        ]
    }
)
