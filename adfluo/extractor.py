from pathlib import Path
from typing import Union, Optional, TextIO, BinaryIO, Literal

from .extraction_graph import ExtractionDAG
from .dataset import DatasetLoader
from .pipeline import ExtractionPipeline

ExtractionOrder = Literal["feature", "sample"]


class Extractor:
    def __init__(self, skip_errors=False):
        self.extraction_DAG = ExtractionDAG()
        self.skip_errors = skip_errors

    def add_extraction(
            self,
            pipeline: ExtractionPipeline,
            skip_errors: Optional[bool] = None,
            drop_on_save: bool = False):

        if not isinstance(pipeline, ExtractionPipeline):
            raise ValueError(f"The pipeline has to be a {ExtractionPipeline} "
                             f"instance")
        # if the fail_on_error parameter is None, defaults to the extractor's
        # fail_on_error
        if skip_errors is None:
            skip_errors = self.skip_errors

        # TODO

    def add_aggregation(self,
                        pipeline,
                        skip_errors: Optional[bool] = None,
                        drop_on_save: bool = False):
        pass

    def extract_to_dict(self,
                        dataset: DatasetLoader,
                        extraction_order: ExtractionOrder = "feature"):
        pass

    def extract_to_csv(self,
                       dataset: DatasetLoader,
                       output_file: Union[str, Path, TextIO],
                       extraction_order: ExtractionOrder = "feature"):
        pass

    def extract_to_pickle(self,
                          dataset: DatasetLoader,
                          output_file: Union[str, Path, BinaryIO],
                          extraction_order: ExtractionOrder = "feature"):
        pass

    def extract_to_df(self,
                      dataset: DatasetLoader,
                      extraction_order: ExtractionOrder = "feature"):
        pass

    def extract_to_hdf5(self,
                        dataset: DatasetLoader,
                        database: Union[str, Path, 'h5py.File'],
                        extraction_order: ExtractionOrder = "feature"):
        pass
