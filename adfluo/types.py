from typing import Union, Any

from typing_extensions import Literal

StorageIndexing = Literal["feature", "sample"]
StorageFormat = Literal["csv", "json", "df", "pickle", "split-pickle", "hdf5"]
FeatureName = str
SampleID = Union[str, int]
SampleData = Any
