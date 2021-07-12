# adfluo

[![Tests](https://github.com/hadware/adfluo/actions/workflows/test.yml/badge.svg)](https://github.com/hadware/adfluo/actions/workflows/test.yml)
[![Documentation](https://github.com/hadware/adfluo/actions/workflows/doc.yml/badge.svg)](https://github.com/hadware/adfluo/actions/workflows/doc.yml)

*adfluo, adfluis, adfluere, adfluxi, adfluxum*

1. to flow on/to/towards/by 
2. to glide/drift quietly

`adfluo` is a Python library for pipeline-oriented feature computation, mainly aimed at tricky 
multimodal datasets that might require a wide range of different features to be computed from.

Adfluo makes your feature extraction code:

- **clean** : it encourages you to outline clearly the steps needed to compute
  a feature as a pipeline of atomic steps
- **data scientist-friendly**: ``adfluo``'s output has a predictable structure,
  ensuring that once you've run the feature extraction, you'll be able to focus a 100%
  on your machine learning/statistics work.
- **efficient** : if different features have computation steps in common, ``adfluo``
  will do its best to compute only what is necessary, without any extra configuration.
- **reusable**: By separating the input data from the feature computation logic,
  you'll easily be able to reuse an existing extraction pipeline on another dataset, 
  or use another extraction pipeline on the same dataset.
- **sample-oriented**: extraction is 
  
## Installation

Adfluo is available on Pypi, and has no special dependencies, a simple

```shell
pip install adfluo
```

will do.

## Example

```python
import random

# Defining our dataset as a list of dicts
my_dataset = [
    {"numbers" : [random.randint(1, 20) for j in range(50)]
     "idx": i}
  for i in range(20)
]
```

## 