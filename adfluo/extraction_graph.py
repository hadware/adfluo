from abc import ABCMeta, abstractmethod
from collections import OrderedDict, deque
from typing import List, Dict, Any, Optional, Iterable, Deque, Set, TYPE_CHECKING

from rich.progress import track

from .dataset import DatasetLoader, Sample
from .exceptions import DuplicateSampleError, BadSampleException
from .processors import BatchProcessor, SampleProcessor, SampleInputProcessor, SampleFeatureProcessor, Input
from .types import FeatureName, SampleID, SampleData
from .utils import extraction_policy

if TYPE_CHECKING:
    from .pipeline import ExtractionPipeline


class BaseGraphNode(metaclass=ABCMeta):

    def __init__(self):
        self.children: List['BaseGraphNode'] = []
        self.parents: List['BaseGraphNode'] = []
        self._depth: Optional[int] = None

    @abstractmethod
    def __hash__(self):
        pass

    def __eq__(self, other: 'BaseGraphNode'):
        return hash(self) == hash(other)

    @abstractmethod
    def __str__(self):
        pass

    @property
    def depth(self):
        if self._depth is None:
            self._depth = max(parent.depth for parent in self.parents) + 1
        return self._depth

    @depth.setter
    def depth(self, value: int):
        self._depth = value

    def iter_all_samples(self) -> Iterable[Sample]:
        if not self.parents:
            raise RuntimeError("Cannot retrieve all samples if no parent is set")
        return self.parents[0].iter_all_samples()

    def ancestor_hash(self) -> int:
        parents_hashes = tuple(parent.ancestor_hash() for parent in self.parents)
        return hash((self, *parents_hashes))

    @abstractmethod
    def __getitem__(self, sample: Sample) -> SampleData:
        pass

    def replace_parent(self, old_parent: 'BaseGraphNode',
                       new_parent: 'BaseGraphNode'):
        parent_idx = self.parents.index(old_parent)
        self.parents[parent_idx] = new_parent

    def replace_child(self, old_child: 'BaseGraphNode',
                      new_child: 'BaseGraphNode'):
        child_idx = self.children.index(old_child)
        self.parents[child_idx] = new_child


class CachedNode(BaseGraphNode, metaclass=ABCMeta):
    """An abstract type for nodes that caches data until it has been retrieved
    (or "pulled") by all of its child nodes."""

    def __init__(self):
        super().__init__()
        self._samples_cache: Dict[SampleID, Any] = dict()
        self._samples_cache_hits: Dict[SampleID, int] = dict()
        self._failed_samples: Set[SampleID] = set()

    @abstractmethod
    def compute_sample(self, sample: Sample) -> Any:
        pass

    def from_cache(self, sample: Sample):
        if sample.id in self._failed_samples:
            raise BadSampleException(sample)

        if sample.id in self._samples_cache:
            # retrieving the sample and incrementing the cache hits counter
            cached_output = self._samples_cache[sample.id]
            self._samples_cache_hits[sample.id] += 1
            # if the cache hits equals the number of children, the sample's
            # value can be dropped from the cache
            if self._samples_cache_hits[sample.id] >= len(self.children):
                del self._samples_cache[sample.id]
            return cached_output
        else:
            raise KeyError("Sample not in cache")

    def to_cache(self, sample: Sample, data: SampleData):
        self._samples_cache[sample.id] = data
        self._samples_cache_hits[sample.id] = 1

    def reset_cache(self):
        self._samples_cache = dict()
        self._samples_cache_hits = dict()

    def __getitem__(self, sample: Sample) -> Sample:
        # if node has no children or one child, or if cache is disabled,
        # bypass the cache mechanism
        if len(self.children) <= 1 or extraction_policy.no_cache:
            return self.compute_sample(sample)

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
        super().__init__()
        self.processor = processor

    def __hash__(self):
        return hash((self.__class__, hash(self.processor)))

    def compute_sample(self, sample: Sample) -> Any:
        try:
            parents_output = tuple(node[sample] for node in self.parents)
        except BadSampleException as err:
            self._failed_samples.add(sample.id)
            raise err

        try:
            return self.processor(sample, parents_output)
        except Exception as err:
            if extraction_policy.skip_errors:
                self._failed_samples.add(sample.id)
                raise BadSampleException(sample)
            else:
                raise err

    def __str__(self):
        return str(self.processor)


class BatchProcessorNode(CachedNode):

    def __init__(self, processor: BatchProcessor):
        super().__init__()
        self.processor = processor
        self.has_computed_batch = False
        self.batch_cache: OrderedDict[SampleID, Any] = OrderedDict()

    def __hash__(self):
        return hash((self.__class__, hash(self.processor)))

    def compute_batch(self):
        # TODO: error handling mechanism.
        #  Idea: when the batch compute is wrong, set the whole batch as a bad sample
        for sample in self.iter_all_samples():
            parents_output = tuple(node[sample] for node in self.parents)
            self.batch_cache[sample.id] = parents_output
        if len(self.parents) == 1:
            all_samples_data = [data[0] for data in self.batch_cache.values()]
        else:
            all_samples_data = list(self.batch_cache.values())

        # TODO: check that
        self.processor.full_dataset_process(all_samples_data)
        self.has_computed_batch = True

    def compute_sample(self, sample: Sample) -> Any:
        if not self.has_computed_batch:
            self.compute_batch()
        parents_output = self.batch_cache.pop(sample.id)
        return self.processor(sample, parents_output)

    def __str__(self):
        return str(self.processor)


class FeatureNode(SampleProcessorNode):
    """Doesn't do any processing, just here as a passthrough node from
    which to pull samples for a specific feature"""

    processor: SampleFeatureProcessor

    @property
    def feature_name(self) -> str:
        return self.processor.feat_name

    def compute_sample(self, sample: Sample) -> Any:
        if not self.parents:
            raise RuntimeError(f"No parents for feature node for feature "
                               f"{self.processor.feat_name}. "
                               f"Node has no parents.")
        return super().compute_sample(sample)

    def ancestor_hash(self) -> int:
        # TODO: document
        return hash(self)


class InputNode(SampleProcessorNode):
    # TODO: doc
    processor: SampleInputProcessor

    def __init__(self, processor: SampleProcessor, is_feat: bool = False):
        super().__init__(processor)
        self.is_feat = is_feat

    @property
    def data_name(self) -> str:
        return self.processor.data_name

    def ancestor_hash(self) -> int:
        return hash(self)


class RootNode(BaseGraphNode):

    def __init__(self):
        super().__init__()
        self.children: List[InputNode] = []
        self.parents = None
        self._loader: Optional[DatasetLoader] = None
        self._depth = 0

    def __hash__(self):
        return hash(self.__class__)

    def set_loader(self, loader: DatasetLoader):
        self._loader = loader

    def iter_all_samples(self) -> Iterable[Sample]:
        return iter(self._loader)

    def __getitem__(self, sample: Sample) -> Sample:
        return sample

    def __str__(self):
        return "Root"

    def ancestor_hash(self) -> int:
        return hash(self)


class ExtractionDAG:
    """
    A DAG that stores the computation graph for all extracted features.
    It has a unique root node, to which all ``InputNode``'s are connected.
    The leaves of this DAG are all ``FeatureNode``.

    The extraction DAG works in "pull" mode: a sample is given to a ``FeatureNode``,
    which will ask its parent node for their output for that one sample, and then
    that node will then in turn ask its parent node (and so on recursively), and then
    run its computation.
    """

    def __init__(self):
        # stores all the processing (input, feature and processor) nodes from
        # the dag
        self.nodes: List[BaseGraphNode] = list()
        # stores only the feature nodes
        self.feature_nodes: Dict[str, FeatureNode] = dict()
        # one and only root from the DAG
        self.root_node: RootNode = RootNode()
        self._loader: Optional[DatasetLoader] = None
        self._needs_dependency_solving = False
        self._features_order: Optional[List[FeatureName]] = None

    @property
    def features(self) -> Set[str]:
        return set(self.feature_nodes.keys())

    @property
    def inputs(self) -> Set[str]:
        return set(input_node.data_name for input_node in self.root_node.children)

    def genealogical_search(self, searched_node: BaseGraphNode) -> Optional[BaseGraphNode]:
        """Search the DAG for a node that is the same node and has the same
        ancestry as the searched node. If nothing is found, returns None"""
        for dag_node in self.nodes:
            if dag_node.ancestor_hash() == searched_node.ancestor_hash():
                return dag_node
        return None

    def add_pipeline(self, pipeline: 'ExtractionPipeline'):
        feature_nodes: List[FeatureNode] = pipeline.outputs
        nodes_stack: Deque[SampleProcessorNode] = deque(feature_nodes)
        # registering feature nodes (and checking that they're not already present)
        for feat_node in feature_nodes:
            # TODO : better error
            assert feat_node.processor.feat_name not in self.feature_nodes
            self.feature_nodes[feat_node.processor.feat_name] = feat_node

        # algorithm outline:
        # stack = list(feature leafs)
        # for node in stack:
        # - pop it from the stack
        # - check if parent nodes' hash is found somewhere in the tree
        # - if parent node hash is found, connect current node to DAG node
        # - else, add parent nodes to stack
        while nodes_stack:
            node = nodes_stack.pop()
            # TODO: document this condition
            if isinstance(node, FeatureNode) and not node.parents:
                node = InputNode(Input(node.feature_name), is_feat=True)

            self.nodes.append(node)
            # an input node has to be directly connected to the root node
            # NOTE: if an input node is put on the stack, it means that this
            # particular input node wasn't already present as a rootnode's child
            if isinstance(node, InputNode):
                assert node.data_name not in self.inputs
                node.parents = [self.root_node]
                self.root_node.children.append(node)
                continue

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
        `FeatureNode` and disconnects them from the root node."""
        if not self._needs_dependency_solving:
            return

        root_children = self.root_node.children
        for node in list(root_children):
            feature_name = node.processor.data_name
            feature_node = self.feature_nodes.get(feature_name)

            # if this input node isn't a feature, skip
            if feature_node is None:
                # It's a feature, yet no feature was found in the graph...
                # this a problem
                if node.is_feat:
                    raise ValueError(f"No matching feature in graph for"
                                     f" input name '{node.data_name}'")
                else:
                    continue

            # remove that input node and link its children to a feature node,
            # that will act as a cache
            for child_node in node.children:
                child_node.replace_parent(node, feature_node)
            # removing the input node from the root node's children
            root_children.remove(node)
            self.nodes.remove(node)

        self._needs_dependency_solving = False

    def prune_features(self,
                       keep_only: Optional[Iterable[str]] = None,
                       remove: Optional[Iterable[str]] = None):
        """Removing features from the DAG (by specifying either the ones that should be
        removed on the ones that should be kept). This is used to optimize the extraction
        when only certain features are needed."""

        assert bool(keep_only) != bool(remove)
        if keep_only is not None:
            kept_features = self.features & set(keep_only)
            removed_features = self.features - kept_features
        else:
            removed_features = self.features & set(remove)
            kept_features = self.features - removed_features

        # sanity measure
        self.solve_dependencies()

        # building the initial stack with all the leaf feature nodes
        # (features that have children are omitted, they might be useful for
        #  some kept features)
        stack: List[BaseGraphNode] = [self.feature_nodes[feat] for feat in removed_features
                                      if len(self.feature_nodes[feat].children) == 0]

        # - all the nodes that don't have any more children are removed from the DAG
        # - if a node is a feature node that shouldn't be removed, it's skipped
        while stack:
            node = stack.pop()
            if isinstance(node, FeatureNode) and node.feature_name in kept_features:
                continue

            self.nodes.remove(node)
            for parent in node.parents:
                parent.children.remove(node)
                if not parent.children:
                    stack.append(parent)

        # once everything has been cleaned, removing the features from the registry:
        for feat in removed_features:
            del self.feature_nodes[feat]

    def compute_feature_order(self):
        # sorting feature node by increasing depth
        sorted_feature_nodes = sorted(self.feature_nodes.values(),
                                      key=lambda node: node.depth)
        self._features_order = [node.processor.feature
                                for node in sorted_feature_nodes]

    def set_loader(self, loader: DatasetLoader):
        self._loader = loader
        self.root_node.set_loader(loader)

    def extract_feature_wise(self, feature_name: str, show_progress: bool) \
            -> Dict[SampleID, Any]:
        """Extract a feature for all samples"""
        self.solve_dependencies()

        feat_dict = {}
        feat_node = self.feature_nodes[feature_name]

        sample_ids = set()
        for sample in track(self._loader,
                            description=feature_name,
                            disable=not show_progress):
            if sample.id in sample_ids:
                raise DuplicateSampleError(sample.id)
            sample_ids.add(sample.id)

            try:
                feat_dict[sample.id] = feat_node[sample]
            except BadSampleException:
                pass
        return feat_dict

    def extract_sample_wise(self, sample: Sample, show_progress: bool) \
            -> Dict[FeatureName, Any]:
        """Extract all features for a unique sample"""
        self.solve_dependencies()

        feat_dict = {}
        for feature_name, feature_node in track(self.feature_nodes.items(),
                                                description=sample.id,
                                                disable=not show_progress):
            try:
                feat_dict[feature_name] = feature_node[sample]
            except BadSampleException:
                pass
        return feat_dict
