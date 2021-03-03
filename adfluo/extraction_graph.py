from abc import ABCMeta, abstractmethod
from collections import OrderedDict, deque
from typing import List, Dict, Any, Optional, Iterable, Deque

from .loader import DatasetLoader
from .pipeline import ExtractionPipeline
from .processors import BatchProcessor, SampleProcessor, SampleInputProcessor, SampleFeatureProcessor
from .samples import Sample

SampleID = str
FeatureName = str
SampleData = Any


class BaseGraphNode(metaclass=ABCMeta):
    children: List['BaseGraphNode'] = []
    parents: List['BaseGraphNode'] = []

    @abstractmethod
    def __hash__(self):
        pass

    def __eq__(self, other: 'BaseGraphNode'):
        return hash(self) == hash(other)

    def iter_all_samples(self) -> Iterable[Sample]:
        return self.parents[0].iter_all_samples()

    def ancestor_hash(self) -> float:
        parents_hashes = tuple(parent.ancestor_hash() for parent in self.parents)
        return hash((self, *parents_hashes))

    @abstractmethod
    def __getitem__(self, sample: Sample) -> SampleData:
        pass

    def replace_parent(self, old_parent: 'BaseGraphNode',
                       new_parent: 'BaseGraphNode'):
        parent_idx = self.parents.index(old_parent)
        self.parents[parent_idx] = new_parent


class CachedNode(BaseGraphNode, metaclass=ABCMeta):
    """An abstract type for nodes that caches data until it has been retrieved
    (or "pulled") by all of its child nodes."""

    _samples_cache: Dict[Sample, Any] = dict()
    _samples_cache_hits: Dict[Sample, int] = dict()
    _fail_on_error: bool = False

    @abstractmethod
    def compute_sample(self, sample: Sample) -> Any:
        pass

    def update_fail_on_error(self, fail_on_error: bool):
        self._fail_on_error = self._fail_on_error or fail_on_error

    def from_cache(self, sample: Sample):
        if sample in self._samples_cache:
            # retrieving the sample and incrementing the cache hits counter
            cached_output = self._samples_cache[sample]
            self._samples_cache_hits[sample] += 1
            # if the cache hits equals the number of children, the sample's
            # value can be dropped from the cache
            if self._samples_cache_hits[sample] >= len(self.children):
                del self._samples_cache[sample]
            return cached_output
        else:
            raise KeyError("Sample not in cache")

    def to_cache(self, sample: Sample, data: SampleData):
        self._samples_cache[sample] = data
        self._samples_cache_hits[sample] = 1

    def reset_cache(self):
        self._samples_cache = {}
        self._samples_cache_hits = {}

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

    def __init__(self, processor: SampleProcessor):
        self.processor = processor

    def compute_sample(self, sample: Sample) -> Any:
        parents_output = tuple(node[sample] for node in self.parents)
        return self.processor(sample, parents_output, fail_on_error=True)


class BatchProcessorNode(CachedNode):

    def __init__(self, processor: BatchProcessor):
        self.processor = processor
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


class FeatureNode(SampleProcessorNode):
    """Doesn't do any processing, just here as a passthrough node from
    which to pull samples for a specific feature"""

    processor: SampleFeatureProcessor

    def ancestor_hash(self) -> float:
        # TODO: document
        return hash(self)


class InputNode(SampleProcessorNode):
    # TODO: doc
    processor: SampleInputProcessor


class RootNode(BaseGraphNode):
    children: List[InputNode] = []
    parents = None

    def __init__(self):
        self._loader: Optional[DatasetLoader] = None

    def __hash__(self):
        return hash(self.__class__)

    def set_loader(self, loader: DatasetLoader):
        self._loader = loader

    def iter_all_samples(self) -> Iterable[Sample]:
        return iter(self._loader)

    def __getitem__(self, sample: Sample) -> Sample:
        return sample


class ExtractionDAG:

    def __init__(self):
        # stores all the processing (intput, feature and processor) nodes from
        # the dag
        self.nodes: List[BaseGraphNode] = list()
        # stores only the feature nodes
        self.feature_nodes: Dict[str, FeatureNode] = dict()
        # one and only root from the DAG
        self.root_node: RootNode = RootNode()
        self._loader: Optional[DatasetLoader] = None
        self._needs_dependency_solving = False

    def genealogical_search(self, searched_node: BaseGraphNode) -> Optional[BaseGraphNode]:
        """Search the DAG for a node that is the same node and has the same
        ancestry as the searched node. If nothing is found, returns None"""
        for dag_node in self.nodes:
            if dag_node.ancestor_hash() == searched_node.ancestor_hash():
                return dag_node
        return None

    def add_pipeline(self, pipeline: ExtractionPipeline):
        feature_nodes = pipeline.pipeline_dag.output_nodes
        nodes_stack: Deque[SampleProcessorNode] = deque(feature_nodes)
        # registering feature nodes (and checking that they're not already present)
        for feat_node in feature_nodes:
            assert feat_node.processor.feat_name not in self.feature_nodes
            self.feature_nodes[feat_node.processor.feat_name] = feat_node

        # algorithm outline:
        # stack = list(feature leafs)
        # for node in stack:
        # - pop it from the stack
        # - check if parent nodes hash is found somewhere in the tree
        # - if parent node hash is found, connect current node to DAG node
        # - else, add parent node to stack
        while nodes_stack:
            node = nodes_stack.pop()
            # an input node cannot be added to the
            if isinstance(node.processor, SampleInputProcessor):
                node.parents = [self.root_node]
                continue

            self.nodes.append(node)
            for node_parent in list(node.parents):
                dag_node = self.genealogical_search(node_parent)
                if dag_node is not None:
                    # replace current parent with dag parent
                    node.replace_parent(node_parent, dag_node)
                    # add the current node as a child to the dag parent
                    dag_node.children.append(node)
                else:
                    nodes_stack.appendleft(node_parent)

        self._needs_dependency_solving = True

    def solve_dependencies(self):
        """Connects inputs that are actually features to the corresponding
        `FeatureNode`"""
        root_children = self.root_node.children
        for input_node in list(root_children):
            feature_node = self.feature_nodes.get(input_node.processor.input)
            # if this input node isn't a feature, skip
            if feature_node is None:
                continue
            # rIemove that input node and link its children to a feature node,
            # that will act as a cache
            for child_node in input_node.children:
                child_node.replace_parent(input_node, feature_node)
            # removing the input node from the root node's children
            root_children.remove(input_node)

    def remove_passthrough(self):
        pass  # TODO

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
