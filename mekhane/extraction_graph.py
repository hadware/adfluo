from abc import ABCMeta, abstractmethod
from collections import OrderedDict
from typing import List, Dict, Tuple, Any, Optional, Iterable

from mekhane.loader import DatasetLoader
from mekhane.processors import BaseProcessor, BatchProcessor, SampleProcessor
from mekhane.samples import Sample

SampleID = str
FeatureName = str
SampleData = Any


class GraphNode(metaclass=ABCMeta):
    children: List['GraphNode'] = []
    parents: List['GraphNode'] = []

    @abstractmethod
    def __hash__(self):
        pass

    def iter_all_samples(self) -> Iterable[Sample]:
        return self.parents[0].iter_all_samples()

    @abstractmethod
    def __getitem__(self, sample: Sample) -> SampleData:
        pass


class CachedNode(GraphNode, metaclass=ABCMeta):
    """An abstract type for nodes that caches data until it has been retrieved
    (or "pulled") by all of its child nodes."""

    _samples_cache: Dict[Sample, Any] = dict()
    _samples_call_count: Dict[Sample, int] = dict()
    _fail_on_error: bool = False

    @abstractmethod
    def compute_sample(self, sample: Sample) -> Any:
        pass

    def update_fail_on_error(self, fail_on_error: bool):
        self._fail_on_error = self._fail_on_error or fail_on_error

    def from_cache(self, sample: Sample):
        if sample in self._samples_cache:
            cached_output = self._samples_cache[sample]
            self._samples_call_count[sample] -= 1
            if self._samples_call_count[sample] <= 0:
                del self._samples_cache[sample]
            return cached_output
        else:
            raise KeyError("Sample not in cache")

    def to_cache(self, sample: Sample, data: SampleData):
        self._samples_cache[sample] = data
        self._samples_call_count[sample] = len(self.children) - 1

    def __getitem__(self, sample: Sample) -> SampleData:
        try:
            return self.from_cache(sample)
        except KeyError:
            sample_data = self.compute_sample(sample)
            self.to_cache(sample, sample_data)
            return sample_data


class SampleProcessorNode(CachedNode):
    """Wraps a processor. If it has several child node, it's able to cache
    the result of its processor for each sample."""

    def __init__(self, parents: List[GraphNode], processor: SampleProcessor):
        self.processor = processor
        self.parents = parents

    def compute_sample(self, sample: Sample) -> Any:
        parents_output = tuple(node[sample] for node in self.parents)
        return self.processor(sample, parents_output, fail_on_error=True)


class BatchProcessorNode(CachedNode):

    def __init__(self, parents: List[GraphNode], processor: BatchProcessor):
        self.processor = processor
        self.parents = parents
        self.has_computed_batch = False
        self.batch_cache: OrderedDict[Sample, Any] = OrderedDict()

    def compute_batch(self):
        for sample in self.iter_all_samples():
            parents_output = tuple(node[sample] for node in self.parents)
            self.batch_cache[sample] = parents_output
        if len(self.parents) == 1:
            all_samples_data = [data[0] for data in self.batch_cache.values()]
        else:
            all_samples_data = list(self.batch_cache.values())
        self.processor.full_dataset_process(list(self.batch_cache.keys()),
                                            all_samples_data)
        self.has_computed_batch = True

    def compute_sample(self, sample: Sample) -> Any:
        if not self.has_computed_batch:
            self.compute_batch()
        parents_output = self.batch_cache.pop(sample)
        return self.processor(sample, parents_output, fail_on_error=True)


class SampleDataNode(CachedNode):
    """Caches each sample's data retrieved from the dataset (to prevent
    it from being recomputed if it's costly.)"""

    def __init__(self, root_node: 'RootNode', feat_input: str):
        self.feat_input = feat_input
        self.parents = [root_node]

    def __hash__(self):
        return hash(self.feat_input)

    def compute_sample(self, sample: Sample) -> Any:
        return sample.get_data(self.feat_input)


class RootNode(GraphNode):

    def __init__(self):
        self._loader: Optional[DatasetLoader] = None

    def set_loader(self, loader: DatasetLoader):
        self._loader = loader

    def iter_all_samples(self) -> Iterable[Sample]:
        return iter(self._loader)

    def __getitem__(self, sample: Sample) -> Sample:
        return sample


class FeatureNode(GraphNode):
    """Doesn't do any processing, just here as a passthrough node from
    which to pull samples for a specific feature"""

    def __init__(self, parent: GraphNode, feature_name: str):
        self.parents = [parent]
        self.feature_name = feature_name
        self.children = []

    def __hash__(self):
        return hash(self.feature_name)

    def __getitem__(self, sample: Sample) -> Any:
        return self.parents[0][sample]


class ExtractionDAG:

    def __init__(self):
        self.feature_nodes: Dict[str, FeatureNode] = dict()
        self.root_node: RootNode = RootNode()
        self._loader: Optional[DatasetLoader] = None

    def set_loader(self, loader: DatasetLoader):
        self._loader = loader
        self.root_node.set_loader(loader)

    def extract_feature_wise(self, feature_name: str) -> Dict[SampleID, Any]:
        """Extract a feature for all samples"""
        feat_dict = {}
        feat_node = self.feature_nodes[feature_name]
        for sample in self._loader:
            feat_dict[sample.id] = feat_node[sample]
        return feat_dict

    def extract_sample_wise(self, sample: Sample) -> Dict[FeatureName, Any]:
        """Extract all features for a unique sample"""
        feat_dict = {}
        for feature_name, feature_node in self.feature_nodes.items():
            feat_dict[feature_name] = feature_node[sample]
        return feat_dict
