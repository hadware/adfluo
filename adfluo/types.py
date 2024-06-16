from typing import Any, Callable, Iterable, Literal

StorageIndexing = Literal["feature", "sample"]
StorageFormat = Literal["csv", "json", "df", "pickle", "split-pickle", "hdf5"]
FeatureName = str
SampleID = str | int
SampleData = Any
ProgressIterator = Callable[[Iterable], Iterable]
