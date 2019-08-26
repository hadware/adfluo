import pickle
from abc import ABC, abstractmethod
from typing import Iterable

from pathlib import Path

from .samples import Sample


class DatasetLoader(ABC):
    """
    Child classes of this class should take care of loading a dataset
    and formatting it to samples, then storing it into the sample
    attribute.
    """

    @property
    @abstractmethod
    def samples(self) -> Iterable[Sample]:
        raise NotImplementedError()

    @abstractmethod
    def __len__(self):
        raise NotImplementedError()

    def __iter__(self) -> Iterable[Sample]:
        return iter(self.samples)

    def save(self,
             output_filepath: Path,
             method="pickle",
             filename: str = None):
        if method == "pickle":
            pkled_dict = {}
            for sample in self.samples:
                pkled_dict[sample.id] = sample.to_pickle()
            with open(str(output_filepath / Path("extraction.pckl")),
                      "wb") as pklfile:
                pickle.dump(pkled_dict, pklfile)
        elif method == "h5":
            # TODO: figure out of how to properly store in h5
            with h5py.File(str(output_filepath), "w") as h5_file:
                for sample in self.samples:
                    sample.to_h5(h5_file)

    def save_pickle(self, filepath: Path):
        pkled_dict = {}
        for sample in self.samples:
            pkled_dict[sample.id] = sample.to_pickle()
        with open(str(filepath), "wb") as pklfile:
            pickle.dump(pkled_dict, pklfile)
