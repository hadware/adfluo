from csv import Dialect
from pathlib import Path
from typing import Union, Optional, TextIO, BinaryIO, Literal, TYPE_CHECKING, List, Dict, Set

from .extraction_graph import ExtractionDAG, FeatureName, FeatureNode, DuplicateSampleError
from .dataset import DatasetLoader, Sample, ListLoader
from .pipeline import ExtractionPipeline
from .storage import BaseStorage, StorageIndexing, CSVStorage, PickleStorage, DataFrameStorage
from .utils import extraction_policy, logger

ExtractionOrder = Literal["feature", "sample"]
Dataset = Union[DatasetLoader, List[Dict], List[Sample]]

if TYPE_CHECKING:
    try:
        import pandas as pd
    except ImportError:
        pass


class Extractor:
    def __init__(self, skip_errors=False, show_progress=True):
        self.extraction_DAG = ExtractionDAG()
        self.skip_errors = skip_errors
        self.show_progress = show_progress
        self.dropped_features: Set[FeatureName] = set()

    def add_extraction(
            self,
            pipeline: ExtractionPipeline,
            drop_on_save: bool = False):

        if not isinstance(pipeline, ExtractionPipeline):
            raise ValueError(f"The pipeline has to be an {ExtractionPipeline} "
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

        self.extraction_DAG.set_loader(dataset)
        if extraction_order == "feature":
            for feature_name in self.extraction_DAG.feature_nodes:
                logger.info(f"Extracting feature {feature_name}")
                output_data = self.extraction_DAG.extract_feature_wise(feature_name,
                                                                       self.show_progress)
                if feature_name in self.dropped_features:
                    continue

                storage.store_feat(feature_name, output_data)
        else:
            sample_ids = set()
            for sample in dataset:
                if sample.id in sample_ids:
                    raise DuplicateSampleError(sample.id)
                sample_ids.add(sample.id)

                logger.info(f"Extracting features for sample {sample.id}")
                output_data = self.extraction_DAG.extract_sample_wise(sample,
                                                                      self.show_progress)

                # dropping "dropped on save" features
                for feat_name in self.dropped_features:
                    if feat_name in output_data:
                        del output_data[feat_name]

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
        extraction_policy.skip_errors = self.skip_errors
        extraction_policy.no_cache = no_caching
        if isinstance(output_file, (Path, str)):
            csv_file = open(output_file, "w")
        else:
            csv_file = output_file

        storage = CSVStorage(storage_indexing, csv_file, csv_dialect)
        self._extract(dataset, extraction_order, storage)

        if isinstance(output_file, (Path, str)):
            csv_file.close()

    def extract_to_pickle(self,
                          dataset: Dataset,
                          output_file: Union[str, Path, BinaryIO],
                          extraction_order: ExtractionOrder = "feature",
                          storage_indexing: StorageIndexing = "sample",
                          no_caching: bool = False):
        extraction_policy.skip_errors = self.skip_errors
        extraction_policy.no_cache = no_caching
        if isinstance(output_file, (Path, str)):
            pickle_file = open(output_file, "wb")
        else:
            pickle_file = output_file

        storage = PickleStorage(storage_indexing, pickle_file)
        self._extract(dataset, extraction_order, storage)

        if isinstance(output_file, (Path, str)):
            pickle_file.close()

    def extract_to_pickle_files(self,
                                dataset: Dataset,
                                output_folder: Union[str, Path],
                                extraction_order: ExtractionOrder = "feature",
                                storage_indexing: StorageIndexing = "sample",
                                no_caching: bool = False):
        pass  # to stream, indexing must be the same as sample order

    def extract_to_df(self,
                      dataset: Dataset,
                      extraction_order: ExtractionOrder = "feature",
                      storage_indexing: StorageIndexing = "sample",
                      no_caching: bool = False) -> 'pd.DataFrame':
        extraction_policy.skip_errors = self.skip_errors
        extraction_policy.no_cache = no_caching
        storage = DataFrameStorage(storage_indexing)
        self._extract(dataset, extraction_order, storage)
        return storage.get_value()

    def extract_to_hdf5(self,
                        dataset: Dataset,
                        database: Union[str, Path, 'h5py.File'],
                        extraction_order: ExtractionOrder = "feature",
                        storage_indexing: StorageIndexing = "sample",
                        no_caching: bool = False):
        pass  # to stream, indexing must be the same as sample order
