from csv import Dialect
from pathlib import Path
from typing import Union, Optional, TextIO, BinaryIO, Literal, TYPE_CHECKING, List, Dict, Set

from .extraction_graph import ExtractionDAG, FeatureName, FeatureNode
from .dataset import DatasetLoader, Sample, ListLoader
from .pipeline import ExtractionPipeline
from .storage import BaseStorage, StorageIndexing
from .utils import extraction_policy

ExtractionOrder = Literal["feature", "sample"]
Dataset = Union[DatasetLoader, List[Dict], List[Sample]]

if TYPE_CHECKING:
    try:
        import pandas as pd
    except ImportError:
        pass


class Extractor:
    def __init__(self, skip_errors=False):
        self.extraction_DAG = ExtractionDAG()
        self.skip_errors = skip_errors
        self.dropped_features: Set[FeatureName] = set()

    def add_extraction(
            self,
            pipeline: ExtractionPipeline,
            drop_on_save: bool = False):

        if not isinstance(pipeline, ExtractionPipeline):
            raise ValueError(f"The pipeline has to be a {ExtractionPipeline} "
                             f"instance")
        pipeline.check()
        self.extraction_DAG.add_pipeline(pipeline)

        if drop_on_save:
            for feat_node in pipeline.outputs:
                feat_node: FeatureNode
                self.dropped_features.add(feat_node.processor.feature)

    def add_aggregation(self,
                        pipeline,
                        skip_errors: Optional[bool] = None,
                        drop_on_save: bool = False):
        raise NotImplementedError()

    def _extract(self,
                 dataset: Dataset,
                 extraction_order: ExtractionOrder,
                 storage: BaseStorage):
        assert extraction_order in ("sample", "feature")
        if isinstance(dataset, list):
            dataset = ListLoader(dataset)
        # TODO: add logging somewhere?
        self.extraction_DAG.set_loader(dataset)
        if extraction_order == "feature":
            for feature_name in self.extraction_DAG.feature_nodes:
                output_data = self.extraction_DAG.extract_feature_wise(feature_name)
                storage.store_feat(feature_name, output_data)
        else:
            for sample in dataset:
                output_data = self.extraction_DAG.extract_sample_wise(sample)
                storage.store_sample(sample, output_data)

    def extract_to_dict(self,
                        dataset: Dataset,
                        extraction_order: ExtractionOrder = "feature",
                        storage_indexing: StorageIndexing = "sample",
                        no_caching: bool = False):
        extraction_policy.skip_errors = self.skip_errors
        extraction_policy.no_cache = no_caching
        storage = BaseStorage(storage_indexing)
        self._extract(dataset, extraction_order, storage)
        return storage.get_value()

    def extract_to_csv(self,
                       dataset: Dataset,
                       output_file: Union[str, Path, TextIO],
                       extraction_order: ExtractionOrder = "feature",
                       storage_indexing: StorageIndexing = "sample",
                       no_caching: bool = False,
                       csv_dialect: Optional[Dialect] = None):
        pass

    def extract_to_pickle(self,
                          dataset: Dataset,
                          output_file: Union[str, Path, BinaryIO],
                          extraction_order: ExtractionOrder = "feature",
                          storage_indexing: StorageIndexing = "sample",
                          no_caching: bool = False):
        pass

    def extract_to_df(self,
                      dataset: Dataset,
                      extraction_order: ExtractionOrder = "feature",
                      storage_indexing: StorageIndexing = "sample",
                      no_caching: bool = False) -> 'pd.DataFrame':
        pass

    def extract_to_hdf5(self,
                        dataset: Dataset,
                        database: Union[str, Path, 'h5py.File'],
                        extraction_order: ExtractionOrder = "feature",
                        storage_indexing: StorageIndexing = "sample",
                        no_caching: bool = False):
        pass
