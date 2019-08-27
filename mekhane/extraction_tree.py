from itertools import cycle

from treelib import Tree, Node
from typing import List, Generator, Any, Iterable, Tuple, Dict

from .processors import BaseProcessor
from .samples import Sample


class BackReferencedNodeMixin(Node):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.parent_node: Node = None

    def set_parent_node(self, tree: 'ProcessorsTree'):
        """The parent node reference has to be set "by hand" as it is not
        set during the build time of the tree."""
        if not self.is_root():
            self.parent_node = tree[self.bpointer]


class ProcessorNode(BackReferencedNodeMixin, Node):
    """A processor node basically wraps a processor, and iterating
    upon that node will lazilly call the processor on the current sample
    being 'pulled' from the node. Nodes that have several child nodes
    cache the processed samples once the first child node has 'pulled'
    them (to prevent reprocessing them in future interations)"""
    def __init__(self, processor: BaseProcessor, fail_on_error: bool):
        super().__init__(tag=repr(processor))
        self.proc = processor
        self.fail_on_error = fail_on_error
        self.sample_iter = None
        self.samples_cache: Dict[Sample, Any] = {}
        self.samples_call: Dict[Sample, int] = {}

    def __iter__(self):
        # Sample iter is the iterator of the parent node, wrapped
        # by the own node's sample processor
        if self.sample_iter is None:
            parent_it = iter(self.parent_node)
            self.sample_iter = self.proc(parent_it,
                                         fail_on_error=self.fail_on_error)

        # if the child count is 1, no caching is needed
        if len(self.fpointer) == 1:
            for sample_couple in self.sample_iter:
                yield sample_couple
        else:
            # processing only once, and caching the samples
            if self.iter_call == 0:
                self.iter_call += 1
                self.samples_cache = []
                for sample_couple in self.sample_iter:
                    self.samples_cache.append(sample_couple)
                    yield sample_couple

            # until (not including) the last call on __iter__, we just
            # output the cached nodes in the same order they are stored
            elif 0 < self.iter_call < len(self.fpointer) - 1:
                self.iter_call += 1
                for sample_couple in self.samples_cache:
                    yield sample_couple

            # at the last call, we empty the cache (setting it to None,
            # which lets the garbage collector do its job)
            else:
                samples_it = iter(self.samples_cache)
                self.samples_cache = None  # freeing the cache
                for sample_couple in samples_it:
                    yield sample_couple

    def __call__(self, sample: Sample):
        if sample in self.samples_cache:
            cached_output = self.samples_cache[sample]
            self.samples_call[sample] -= 1
            if self.samples_call[sample] <= 0:
                del self.samples_cache[sample]
            return cached_output
        else:
            output = self.proc()
        # TODO : process output from parent node. If node has several children,
        #  cache result for that one sample using the sample_id as a key
        #  the cached result should be cleaned when all the child have retrieved the
        #  processed result


class FeatureLeaf(Node):
    """Doesn't do any processing, just here as a special kind of node from
    which to pull samples for a specific feature"""
    def __init__(self, feature: str):
        super().__init__(tag=feature, identifier=feature)
        self.parent_node: ProcessorNode = None

    def set_parent_node(self, tree: 'ProcessorsTree'):
        self.parent_node = tree[self.bpointer]

    def __iter__(self):
        return iter(self.parent_node)

    def __call__(self, sample: Sample):
        return self.parent_node(sample)
        # TODO: should be called upon a sample, and should propagate the call to the parent node


class RootNode(Node):
    """This node is a "passthrough" node that allows to have a unified tree"""
    pass


class FeatureInputNode(Node):
    """This node is a "passthrough" node who's only job is to retrieve the input data from the
    sample in the dataset, and then cache it"""

    def __init__(self, feat_input: str):
        # TODO : check what identifier's purpose is in the treelib doc again
        super().__init__(tag=feat_input, identifier=feat_input)

    def __call__(self, sample: Sample):
        return sample.get_data(self.tag)


class ProcessorsTree(Tree):
    def __init__(self, fail_on_error: bool):
        super().__init__()
        self.fail_on_error = fail_on_error
        self.add_node(RootNode())

    def add_pipeline(self, processors: List[BaseProcessor], feature: str):
        # TODO : the input data name should be passed as well, and the root node
        assert len(processors) > 0
        parent_node = self.get_node(self.root)
        proc_iter = iter(processors)
        # this "matches" the added pipeline with the already present processor
        # tree. It breaks when no matching nodes is found and the new pipeline
        # has to continue into a new branch. `parent_node` is a buffer that
        # stores the current parent node in the tree traversal
        for i, proc in enumerate(proc_iter):
            # this clause is invoked if the tree is empty
            if parent_node is None:
                root_node = ProcessorNode(
                    proc, fail_on_error=self.fail_on_error)
                self.add_node(root_node)
                parent_node = root_node
                break

            # this is just a sanity check that ensure that the first
            # node of the pipeline is the root node of the tree
            if i == 0:
                assert parent_node.proc == proc
                continue

            # checking if any of the child node of the current parent_node
            # is the same as the current node. If none are matching,
            # it means the rest of the nodes are on to "found" their onw
            # branch (the "else" clause)
            for node in self.children(parent_node.identifier):
                if not isinstance(node, FeatureLeaf) and node.proc == proc:
                    parent_node = node
                    break  # breaks the current loop, not the "big" one
            else:  # executed if the previous loop didn't break
                new_node = ProcessorNode(proc, self.fail_on_error)
                self.add_node(new_node, parent=parent_node)
                parent_node = new_node
                break

        # remaining procs not represented in the tree are added as their
        # "own" new branch
        for proc in proc_iter:
            new_node = ProcessorNode(proc, self.fail_on_error)
            self.add_node(new_node, parent=parent_node)
            parent_node = new_node

        # each pipeline and each branch has to end with a feature leaf
        self.add_node(FeatureLeaf(feature), parent=parent_node)

    def set_sample_iter(self, samples_couples: Iterable[Tuple[Sample, Any]]):
        self.get_node(self.root).set_sample_iter(samples_couples)
        for node in self.all_nodes_itr():
            node.set_parent_node(self)

    def get_feature_order(self) -> List[str]:
        feature_leafs = [(node, self.depth(node))
                         for node in self.all_nodes_itr()
                         if isinstance(node, FeatureLeaf)]
        feature_leafs.sort(key=lambda x: x[1])
        return [feat.tag for feat, _ in feature_leafs]

    def __call__(self, feature: str) -> Iterable:
        return iter(self.get_node(feature))
