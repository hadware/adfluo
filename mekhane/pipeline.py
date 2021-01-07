from typing import List, Union, Set
from dataclasses import dataclass

from .extraction_graph import BaseGraphNode, FeatureNode, SampleProcessorNode, BatchProcessorNode
from .processors import BaseProcessor, FunctionWrapperProcessor, SampleInputProcessor, SampleProcessor, BatchProcessor, \
    Input
from .utils import are_consecutive_int

PipelineElement = Union['ExtractionPipeline', BaseProcessor, 'T', 'Feat']

PIPELINE_TYPE_ERROR = "Invalid object in pipeline of type {obj_type}"


class PipelineBuildError(Exception):
    pass


@dataclass(frozen=True)
class Feat:
    feat_name: str

    def __hash__(self):
        return hash(self.feat_name)


class T:
    def __init__(self, *args):
        for arg in args:
            if not isinstance(arg, (Feat, BaseProcessor)):
                raise TypeError(PIPELINE_TYPE_ERROR.format(obj_type=type(args)))
        self.elements = args

    def check(self):
        """Checking that if the tuple contains features or inputs, these
        are placed at the *end* of the tuple"""
        el_it = iter(self.elements)
        for element in el_it:
            if isinstance(element, (Feat, Input)):
                break
        for element in el_it:
            assert isinstance(element, (Feat, Input))

    def __rshift__(self, other: PipelineElement):
        if isinstance(other, (BaseProcessor, T, Feat)):
            return ExtractionPipeline([self, other])
        elif callable(other):
            return ExtractionPipeline([self, FunctionWrapperProcessor(other)])
        elif isinstance(other, ExtractionPipeline):
            other.pipeline.insert(0, self)
        else:
            raise PipelineBuildError(PIPELINE_TYPE_ERROR.format(obj_type=type(other)))


class Branch:
    element: List[BaseGraphNode]

    def __init__(self):
        self.elements: List[BaseGraphNode] = []

    @property
    def head(self):
        return self.elements[-1]


class PipelineDAG:

    def __init__(self):
        self.branches: List['PipelineDAG', Branch] = []
        self.features: Set[FeatureNode] = set()

    @property
    def head(self):
        assert len(self.branches) == 1
        return self.branches[0].head

    @property
    def output_features(self) -> List[Feat]:
        pass

    @property
    def depends_on_features(self) -> List[Feat]:
        pass

    def create_branch(self, input_proc: Input):
        pass

    def add_element_to_branch(self, branch_id: int,
                              element: Union[T, Feat, BaseProcessor]):
        pass

    def add_tuple(self, t: T):
        for i, element in enumerate(t.elements):
            if isinstance(element, (Input, Feat)):
                self.create_branch(element)
            elif isinstance(element, SampleProcessor):
                pass
            elif isinstance(element, T):
                pass

    def join_branches(self, branches_id: List[int], processor: BaseProcessor):
        assert isinstance(processor, SampleProcessor)
        assert len(branches_id) == processor.nb_args
        assert len(branches_id) > 1
        assert are_consecutive_int(branches_id)

        parents = []
        for i in branches_id:
            branch = self.branches[i]
            if isinstance(branch, Branch):
                parents.append()
        parents = [self.branches[i] for i in branches_id]
        if isinstance(processor, BatchProcessor):
            new_node = BatchProcessorNode(parents, processor)
        else:
            new_node = SampleProcessorNode(parents, processor)

        # removing the joined branches heads and replacing them with the new head
        for branch_id in branches_id:
            del self.branches[branch_id]
        self.branches.insert(branches_id[0], new_node)

    def add_pl_element(self, element: Union[T, Feat, BaseProcessor]):
        if isinstance(element, (Feat, BaseProcessor)):
            if len(self.branches) == 0:
                pass
            elif len(self.branches) == 1:
                self.add_element_to_branch(0, element)
            else:
                all_branches = list(range(len(self.branches)))
                self.join_branches(all_branches, element)
        elif isinstance(element, T):
            self.add_tuple(element)
        else:
            raise PipelineBuildError(PIPELINE_TYPE_ERROR.format(obj_type=type(element)))


class ExtractionPipeline:
    """An extraction pipeline is a small DAG that describes the steps
    needed to extract one or several features."""

    def __init__(self,
                 processors: List[Union[BaseProcessor, T]],
                 fail_on_error: bool = True):
        self.pipeline = processors
        self.fail_on_error = fail_on_error

    def __rshift__(self, other: PipelineElement):
        if isinstance(other, (BaseProcessor, T, Feat)):
            self.pipeline.append(other)
        elif callable(other):
            self.pipeline.append(FunctionWrapperProcessor(other))
        elif isinstance(other, ExtractionPipeline):
            self.pipeline += other.pipeline
        else:
            raise PipelineBuildError(PIPELINE_TYPE_ERROR.format(obj_type=type(other)))

    def check(self):
        pass
        # TODO :
        #  * has to end with a feature or a feature tuple
        #  * has to start with inputs, input tuples or feature input
        #  *

    def build_pipeline_tree(self):
        pass


