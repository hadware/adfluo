import warnings
from csv import Dialect
from functools import partial
from itertools import chain
from pathlib import Path
from typing import Union, Optional, TextIO, BinaryIO, TYPE_CHECKING, List, Dict, Set

import sys
from rich.progress import track
from tqdm import tqdm
from typing_extensions import Literal, Any

from .dataset import DatasetLoader, Sample, ListLoader
from .exceptions import DuplicateSampleError
from .extraction_graph import ExtractionDAG, FeatureName, FeatureNode, SampleProcessorNode, AggregatorNode, \
    BaseGraphNode
from .pipeline import ExtractionPipeline
from .storage import BaseStorage, CSVStorage, PickleStorage, DataFrameStorage, JSONStorage, \
    SplitPickleStorage
from .types import StorageIndexing
from .utils import logger, ExtractionPolicy

ExtractionOrder = Literal["feature", "sample"]
Dataset = Union[DatasetLoader, List[Dict], List[Sample]]

if TYPE_CHECKING:
    try:
        import pandas as pd
    except ImportError:
        pass


class Extractor:
    def __init__(self, show_progress=True):
        self.extraction_DAG = ExtractionDAG()
        self.show_progress = show_progress
        self.dropped_features: Set[FeatureName] = set()

    @property
    def hparams(self) -> Set[str]:
        return set(chain.from_iterable(node.processor.hparams for node in self.extraction_DAG.nodes
                                       if isinstance(node, (SampleProcessorNode, AggregatorNode))))

    def set_hparams(self, params: Dict[str, Any]):
        assert set(params.keys()) == self.hparams
        for node in self.extraction_DAG.nodes:
            if isinstance(node, (SampleProcessorNode, AggregatorNode)):
                node.processor.set_hparams(**params)

    def set_extraction_policy(self, no_cache: bool, skip_errors: bool):
        BaseGraphNode.extraction_policy = ExtractionPolicy(skip_errors=skip_errors, no_cache=no_cache)

    def get_progress_iterator(self, caption: str):
        if 'ipykernel' in sys.modules:
            return partial(tqdm, desc=caption, disable=not self.show_progress)
        else:
            return partial(track, description=caption, disable=self.show_progress)

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
                self.dropped_features.add(feat_node.processor.feat_name)

    def extract_aggregations(self, dataset: Dataset):
        """Temporary (?) method to extract aggregations for dataset features that have a
        storage protocol"""
        if self.hparams:
            raise RuntimeError(f"Hyperparameters {', '.join(self.hparams)} should have been set before extraction.")

        if isinstance(dataset, list):
            dataset = ListLoader(dataset)

        self.extraction_DAG.set_loader(dataset)

        output_dict = {}
        for feat_name, feat_value in self.extraction_DAG.extract_dataset_features(self.get_progress_iterator("")):
            feat_node = self.extraction_DAG.dataset_features_nodes[feat_name]
            if feat_node.processor.custom_storage is None:
                output_dict[feat_name] = feat_value
            else:
                feat_node.processor.custom_storage.store_aggregation(feat_name, feat_node())
        return output_dict

    def _extract(self,
                 dataset: Dataset,
                 extraction_order: ExtractionOrder,
                 storage: BaseStorage):
        if self.hparams:
            raise RuntimeError(f"Hyperparameters {', '.join(self.hparams)} still need to be set.")

        assert extraction_order in ("sample", "feature")
        if isinstance(dataset, list):
            dataset = ListLoader(dataset)

        self.extraction_DAG.set_loader(dataset)
        # feature-wise extraction
        if extraction_order == "feature":
            for feature_name, feat_node in self.extraction_DAG.feature_nodes.items():
                logger.info(f"Extracting feature {feature_name}")
                output_data = self.extraction_DAG.extract_feature_wise(feature_name,
                                                                       self.get_progress_iterator(feature_name))
                if feature_name in self.dropped_features:
                    continue

                if feat_node.processor.custom_storage is None:
                    storage.store_feat(feature_name, output_data)
                else:
                    for sample_id, value in output_data.items():
                        feat_node.processor.custom_storage.store(sample_id, feature_name, value)

        else:  # sample-wise extraction
            sample_ids = set()
            for sample in dataset:
                if sample.id in sample_ids:
                    raise DuplicateSampleError(sample.id)
                sample_ids.add(sample.id)

                logger.info(f"Extracting features for sample {sample.id}")
                output_data = self.extraction_DAG.extract_sample_wise(sample,
                                                                      self.get_progress_iterator(sample.id))

                # dropping "dropped on save" features
                for feat_name in self.dropped_features:
                    if feat_name in output_data:
                        del output_data[feat_name]

                # handling custom storage
                for feat_name, value in list(output_data.items()):
                    feat_node = self.extraction_DAG.feature_nodes[feat_name]
                    if feat_node.processor.custom_storage is not None:
                        feat_node.processor.custom_storage.store(sample.id, feat_name, value)
                        del output_data[feat_name]
                storage.store_sample(sample, output_data)

    def extract_to_dict(self,
                        dataset: Dataset,
                        extraction_order: ExtractionOrder = "feature",
                        storage_indexing: StorageIndexing = "sample",
                        no_caching: bool = False,
                        skip_errors: bool = False):
        self.set_extraction_policy(no_caching, skip_errors)

        storage = BaseStorage(storage_indexing)
        self._extract(dataset, extraction_order, storage)
        return storage.get_data()

    def extract_to_csv(self,
                       dataset: Dataset,
                       output_file: Union[str, Path, TextIO],
                       extraction_order: ExtractionOrder = "feature",
                       storage_indexing: StorageIndexing = "sample",
                       no_caching: bool = False,
                       skip_errors: bool = False,
                       csv_dialect: Optional[Dialect] = None):
        self.set_extraction_policy(no_caching, skip_errors)

        if isinstance(output_file, (Path, str)):
            csv_file = open(output_file, "w")
        else:
            csv_file = output_file

        storage = CSVStorage(storage_indexing, csv_file, csv_dialect)
        self._extract(dataset, extraction_order, storage)
        storage.write()

        if isinstance(output_file, (Path, str)):
            csv_file.close()

    def extract_to_pickle(self,
                          dataset: Dataset,
                          output_file: Union[str, Path, BinaryIO],
                          extraction_order: ExtractionOrder = "feature",
                          storage_indexing: StorageIndexing = "sample",
                          no_caching: bool = False,
                          skip_errors: bool = False):
        self.set_extraction_policy(no_caching, skip_errors)

        if isinstance(output_file, (Path, str)):
            pickle_file = open(output_file, "wb")
        else:
            pickle_file = output_file

        storage = PickleStorage(storage_indexing, pickle_file)
        self._extract(dataset, extraction_order, storage)
        storage.write()

        if isinstance(output_file, (Path, str)):
            pickle_file.close()

    def extract_to_json(self,
                        dataset: Dataset,
                        output_file: Union[str, Path, TextIO],
                        extraction_order: ExtractionOrder = "feature",
                        storage_indexing: StorageIndexing = "sample",
                        no_caching: bool = False,
                        skip_errors: bool = False):
        self.set_extraction_policy(no_caching, skip_errors)

        if isinstance(output_file, (Path, str)):
            json_file = open(output_file, "w")
        else:
            json_file = output_file

        storage = JSONStorage(storage_indexing, json_file)
        self._extract(dataset, extraction_order, storage)
        storage.check_samples()
        storage.write()

        if isinstance(output_file, (Path, str)):
            json_file.close()

    def extract_to_pickle_files(self,
                                dataset: Dataset,
                                output_folder: Union[str, Path],
                                extraction_order: ExtractionOrder = "sample",
                                storage_indexing: StorageIndexing = "sample",
                                no_caching: bool = False,
                                skip_errors: bool = False,
                                stream: bool = True):

        self.set_extraction_policy(no_caching, skip_errors)

        if stream:
            assert extraction_order == storage_indexing
        if isinstance(output_folder, str):
            output_folder = Path(output_folder)
        assert output_folder.is_dir()

        storage = SplitPickleStorage(storage_indexing, output_folder, stream)
        self._extract(dataset, extraction_order, storage)
        storage.write()

    def extract_to_df(self,
                      dataset: Dataset,
                      extraction_order: ExtractionOrder = "feature",
                      storage_indexing: StorageIndexing = "sample",
                      no_caching: bool = False,
                      skip_errors: bool = False) -> 'pd.DataFrame':
        self.set_extraction_policy(no_caching, skip_errors)
        storage = DataFrameStorage(storage_indexing)
        self._extract(dataset, extraction_order, storage)
        return storage.get_data()


    def extract_to_hdf5(self,
                        dataset: Dataset,
                        database: Union[str, Path, 'h5py.File'],
                        extraction_order: ExtractionOrder = "sample",
                        storage_indexing: StorageIndexing = "sample",
                        no_caching: bool = False,
                        skip_errors: bool = False):
        raise NotImplementedError()  # to stream, indexing must be the same as sample order

    def _repr_svg_(self):
        from .plots import SVGGraphRenderer
        try:
            return SVGGraphRenderer().render_svg(self.extraction_DAG)
        except ImportError as err:
            warnings.warn(str(err))
