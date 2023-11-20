from abc import ABCMeta, abstractmethod
from collections import deque
from typing import List, Dict, Any, Optional, Iterable, Deque, Set, TYPE_CHECKING, Type

from rich.progress import track

from .cache import BaseCache, SingleValueCache, SampleCache
from .dataset import DatasetLoader, Sample, DictSample
from .exceptions import DuplicateSampleError, BadSampleException, BadAggregationException
from .processors import SampleProcessor, SampleInputProcessor, SampleFeatureProcessor, Input, DatasetFeatureProcessor, \
    DatasetAggregator, DatasetInputProcessor, DSInput
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


class SampleProcessorNode(BaseGraphNode):
    """Wraps a processor. If it has several child node, it's able to cache
    the result of its processor for each sample."""

    default_cache_type: Type[BaseCache] = SampleCache

    def __init__(self, processor: SampleProcessor, cache: Optional[BaseCache] = None):
        super().__init__()
        self.cache = cache if cache is not None else self.default_cache_type()
        self.processor = processor

    def compute_sample(self, sample: Sample) -> Any:
        try:
            parents_output = tuple(node[sample] for node in self.parents)
        except BadSampleException as err:
            self.cache.add_failed_sample(sample)
            raise err

        try:
            return self.processor(sample, parents_output)
        except Exception as err:
            if extraction_policy.skip_errors:
                self.cache.add_failed_sample(sample)
                raise BadSampleException(sample)
            else:
                raise err

    def __getitem__(self, sample: Sample) -> Sample:
        # if node has no children or one child, or if cache is disabled,
        # bypass the cache mechanism
        if len(self.children) <= 1 or extraction_policy.no_cache:
            return self.compute_sample(sample)

        try:
            return self.cache[sample]
        except KeyError:
            sample_data = self.compute_sample(sample)
            self.cache[sample] = sample_data
            return sample_data

    def __hash__(self):
        return hash((self.__class__, hash(self.processor)))

    def __str__(self):
        return str(self.processor)


class AggregatorNode(BaseGraphNode):

    def __init__(self, processor: DatasetAggregator):
        super().__init__()
        self.cache = SingleValueCache()
        self.processor = processor

    def compute_aggregation(self) -> Any:
        all_samples_data = dict()
        for sample in self.iter_all_samples():
            all_samples_data[sample] = tuple(node[sample] for node in self.parents)

        try:
            return self.processor(all_samples_data)
        except Exception as err:
            if extraction_policy.skip_errors:
                raise BadAggregationException()
            else:
                raise err

    def __getitem__(self, sample: Sample) -> Sample:
        # value is always cached as it is just a single value for the whole dataset
        # NOTE : "sample" here isn't actually useful for cache retrieval
        try:
            return self.cache[sample]
        except KeyError:
            sample_data = self.compute_aggregation()
            self.cache[sample] = sample_data
            return sample_data

    def __hash__(self):
        return hash((self.__class__, hash(self.processor)))

    def __str__(self):
        return str(self.processor)


class BaseFeatureNode(SampleProcessorNode):
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


class FeatureNode(BaseFeatureNode):
    """Doesn't do any processing, just here as a passthrough node from
    which to pull samples for a specific feature"""
    processor: SampleFeatureProcessor


class DatasetFeatureNode(BaseFeatureNode):
    """Doesn't do any processing, just here as a passthrough node from
    which pull a dataset feature"""
    default_cache_type = SingleValueCache
    processor: DatasetFeatureProcessor


class BaseInputNode(SampleProcessorNode):
    def __init__(self, processor: SampleProcessor, is_feat: bool = False):
        super().__init__(processor)
        self.is_feat = is_feat

    @property
    def data_name(self) -> str:
        return self.processor.data_name

    def ancestor_hash(self) -> int:
        return hash(self)


class InputNode(BaseInputNode):
    # TODO: doc
    processor: SampleInputProcessor


class DatasetInputNode(BaseInputNode):
    # TODO: doc
    processor: DatasetInputProcessor
    default_cache_type = SingleValueCache


class RootNode(BaseGraphNode):

    def __init__(self):
        super().__init__()
        self.children: List[BaseInputNode] = []
        self.parents = None
        self._loader: Optional[DatasetLoader] = None
        self._depth = 0

    def __hash__(self):
        return hash(self.__class__)

    def set_loader(self, loader: DatasetLoader):
        self._loader = loader
        for child in self.children:
            if isinstance(child, DatasetInputNode):
                child.processor.dataset = loader

    def iter_all_samples(self) -> Iterable[Sample]:
        return iter(self._loader)

    def __getitem__(self, sample: Sample) -> Sample:
        return sample

    def __str__(self):
        return "Root"

    def ancestor_hash(self) -> int:
        # TODO: document
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
        # stores only the feature nodes
        self.dataset_features_nodes: Dict[str, DatasetFeatureNode] = dict()
        # one and only root from the DAG
        self.root_node: RootNode = RootNode()
        self._loader: Optional[DatasetLoader] = None
        self._needs_dependency_solving = False
        self._features_order: Optional[List[FeatureName]] = None

    @property
    def features(self) -> Set[str]:
        return set(self.feature_nodes.keys())

    @property
    def dataset_features(self) -> Set[str]:
        return set(self.dataset_features_nodes.keys())

    @property
    def all_features(self):
        return self.features | self.dataset_features

    @property
    def inputs(self) -> Set[str]:
        return set(input_node.data_name for input_node in self.root_node.children
                   if isinstance(input_node, InputNode))

    @property
    def dataset_inputs(self) -> Set[str]:
        return set(input_node.data_name for input_node in self.root_node.children
                   if isinstance(input_node, DatasetInputNode))

    @property
    def all_inputs(self):
        return self.inputs | self.dataset_inputs

    def genealogical_search(self, searched_node: BaseGraphNode) -> Optional[BaseGraphNode]:
        """Search the DAG for a node that is the same node and has the same
        ancestry as the searched node. If nothing is found, returns None"""
        for dag_node in self.nodes:
            if dag_node.ancestor_hash() == searched_node.ancestor_hash():
                return dag_node
        return None

    def add_pipeline(self, pipeline: 'ExtractionPipeline'):
        # first, checking that the pipeline is right
        pipeline.check()

        feature_nodes: List[BaseFeatureNode] = pipeline.outputs
        nodes_stack: Deque[BaseGraphNode] = deque(feature_nodes)
        # registering feature nodes (and checking that they're not already present)
        for feat_node in feature_nodes:
            # TODO : better error
            # checking that there isn't already a feature named like the ones
            # from this pipeline
            feat_name = feat_node.processor.feat_name
            assert feat_name not in self.all_features, \
                f"Duplicate name for feature name '{feat_name}'"
            if isinstance(feat_node, FeatureNode):
                self.feature_nodes[feat_name] = feat_node
            elif isinstance(feat_node, DatasetFeatureNode):
                self.dataset_features_nodes[feat_name] = feat_node

        # algorithm outline:
        # stack = list(feature leafs)
        # for node in stack:
        # - pop it from the stack
        # - check if parent nodes' hash is found somewhere in the tree
        # - if parent node hash is found, connect current node to DAG node
        # - else, add parent nodes to stack
        while nodes_stack:
            node = nodes_stack.pop()
            # This condition is for nodes that are used as inputs.
            # These will be 'dependency-solved' later on, for now
            # treating them as inputs
            if isinstance(node, BaseFeatureNode) and not node.parents:
                # TODO : adapt this for dual types of inputs
                if isinstance(node, FeatureNode):
                    node = InputNode(Input(node.feature_name), is_feat=True)
                else:
                    node = DatasetInputNode(DSInput(node.feature_name), is_feat=True)


            self.nodes.append(node)
            # an input node has to be directly connected to the root node
            # NOTE: if an input node is put on the stack, it means that this
            # particular input node wasn't already present as a rootnode's children
            if isinstance(node, BaseInputNode):
                assert node.data_name not in self.all_inputs
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
        # TODO: adapt to the DS feats

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
            if isinstance(node, BaseFeatureNode) and node.feature_name in kept_features:
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

    def extract_dataset_features(self, show_progress: bool):
        feat_dict = {}
        fake_sample = DictSample({}, 0)
        for feature_name, feature_node in track(self.dataset_features_nodes.items(),
                                                disable=not show_progress):
            feat_dict[feature_name] = feature_node[fake_sample]
        return feat_dict
