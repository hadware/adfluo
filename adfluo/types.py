from typing import Union, Any

from typing_extensions import Literal

StorageIndexing = Literal["feature", "sample"]
FeatureName = str
SampleID = Union[str, int]
SampleData = Any
