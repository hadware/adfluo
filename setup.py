# Always prefer setuptools over distutils
from setuptools import setup, find_packages
# To use a consistent encoding
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()


setup(
    name='mekhane',
    version='0.1.0',
    description='Pipeline-oriented feature extraction for multimodal datasets',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://gitlab.com/ezhealth/mekhane',
    author='Hadrien Titeux',
    author_email='hadrien.titeux@ens.fr',
    license="MIT",
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Natural Language :: English',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.6'
    ],
    keywords='',
    packages=find_packages(exclude=['docs', 'tests']),
    setup_requires=['pytest-runner', 'setuptools>=38.6.0'],  # >38.6.0 needed for markdown README.md
    tests_require=['pytest'],
)
