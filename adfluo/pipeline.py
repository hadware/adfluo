from abc import ABCMeta, abstractmethod
from collections import deque
from typing import List, Union, Deque, Tuple, Optional

from dataclasses import dataclass

from .extraction_graph import BaseGraphNode, SampleProcessorNode, BatchProcessorNode, FeatureNode
from .processors import BaseProcessor, FunctionWrapperProcessor, SampleProcessor, BatchProcessor, \
    Input, Feat

PipelineElement = Union['ExtractionPipeline', BaseProcessor, 'T', 'Feat']

PIPELINE_TYPE_ERROR = "Invalid object in pipeline of type {obj_type}"


class PipelineBuildError(Exception):
    pass


class T:
    def __init__(self, *args):
        for arg in args:
            if not isinstance(arg, (Feat, BaseProcessor)):
                raise TypeError(PIPELINE_TYPE_ERROR.format(obj_type=type(args)))
        self.elements = args
        self.check()

    def check(self):
        """Checking that if the tuple contains features or inputs, these
        are placed at the *end* of the tuple"""
        el_it = iter(self.elements)
        for element in el_it:
            if isinstance(element, (Feat, Input)):
                break
        for element in el_it:
            assert isinstance(element, (Feat, Input))

    def __len__(self):
        return len(self.elements)

    def __rshift__(self, other: PipelineElement):
        if isinstance(other, (BaseProcessor, T, Feat)):
            return ExtractionPipeline([self, other])
        elif callable(other):
            return ExtractionPipeline([self, FunctionWrapperProcessor(other)])
        elif isinstance(other, ExtractionPipeline):
            other.elements.insert(0, self)
        else:
            raise PipelineBuildError(PIPELINE_TYPE_ERROR.format(obj_type=type(other)))


def wrap_processor(proc: BaseProcessor) -> Union[SampleProcessorNode,
                                                 BatchProcessorNode]:
    if isinstance(proc, Feat):
        return FeatureNode(proc)
    elif isinstance(proc, BatchProcessor):
        return BatchProcessorNode(proc)
    elif isinstance(proc, SampleProcessor):
        return SampleProcessorNode(proc)
    else:
        raise PipelineBuildError(PIPELINE_TYPE_ERROR.format(obj_type=type(proc)))


class BasePipeline(metaclass=ABCMeta):

    @abstractmethod
    @property
    def head(self) -> BaseGraphNode:
        pass

    @abstractmethod
    def append(self, element: BaseProcessor):
        pass


# TODO: change branch API to something simpler
class Branch(BasePipeline):
    element: List[BaseGraphNode]

    def __init__(self):
        self._head: BaseGraphNode = None

    @property
    def head(self):
        return self._head

    @head.setter
    def head(self, node: BaseGraphNode):
        self._head = node

    def append(self, element: BaseProcessor):
        if self._head is None:
            assert isinstance(element, Input)
            self._head = SampleProcessorNode(element)
        else:
            assert not isinstance(element, Input)

            node = wrap_processor(element)
            self._head.children = [node]
            element.parents = [node]
            self._head = node


class PipelineDAG(BasePipeline):

    def __init__(self):
        self.branches: List[BasePipeline] = []
        self.input_nodes: Optional[List[Input]] = None
        self.output_nodes: Optional[List[Feat]] = None

    @property
    def head(self):
        assert len(self.branches) == 1
        return self.branches[0].head

    def walk(self) -> Tuple[List[BaseGraphNode], List[BaseGraphNode]]:
        """Starting from the head of its branches, goes up the DAG to gather
        input and output nodes"""
        self.input_nodes = []
        self.output_nodes = []
        nodes_stack: Deque[BaseGraphNode] = deque()
        for branch in self.branches:
            self.output_nodes.append(branch.head)
            assert isinstance(branch.head, Fea)
            nodes_stack.appendleft(branch.head)

        while nodes_stack:
            # if a node on the stack has parents, put them on the stack, else,
            # it must be an input node, and put it in the input nodes list
            node = nodes_stack.pop()
            if node.parents:
                for parent in node.parents:
                    nodes_stack.appendleft(parent)
            else:
                self.input_nodes.append(node)

    def create_branch(self, input_proc: Union[Feat, Input]):
        assert isinstance(input_proc, (Input, Feat))
        if isinstance(input_proc, Feat):
            input_proc = Input(input_proc.feat_name, is_feat=True)
        new_branch = Branch()
        new_branch.append(input_proc)
        self.branches.append(new_branch)

    def convert_branch_to_dag(self, branch_id):
        new_dag = PipelineDAG()
        new_dag.branches = [self.branches[branch_id]]
        self.branches[branch_id] = new_dag

    def split_branch(self, t: T):
        old_branch = self.branches[0]
        self.branches = []
        for element in t.elements:
            if isinstance(element, BaseProcessor):
                new_node = wrap_processor(element)
                new_node.parents = [old_branch.head]
                new_branch = Branch()
                new_branch.head = new_node
                self.branches.append(new_branch)
            elif isinstance(element, T):
                pipeline_branch = Branch()
                pipeline_branch.head = old_branch.head
                new_pipeline = PipelineDAG()
                new_pipeline.branches = [pipeline_branch]
                new_pipeline.append(element)
                self.branches.append(new_pipeline)
            else:
                # TODO : better error
                raise PipelineBuildError(PIPELINE_TYPE_ERROR.format(obj_type=type(element)))

    def add_tuple(self, t: T):
        assert len(self.branches) <= len(t)
        for i, element in enumerate(t.elements):
            if i < len(self.branches):
                if isinstance(element, SampleProcessor):
                    self.branches[i].append(element)
                elif isinstance(element, T):
                    if isinstance(self.branches[i], Branch):
                        self.convert_branch_to_dag(i)
                    self.branches[i].append(element)
                else:
                    # TODO : better error
                    raise PipelineBuildError(PIPELINE_TYPE_ERROR.format(obj_type=type(element)))
            else:
                # create a new branch
                self.create_branch(element)

    def join_branches(self, processor: BaseProcessor):
        # TODO : better error
        assert len(self.branches) == processor.nb_args

        parents = [branch.head for branch in self.branches]
        new_node = wrap_processor(processor)
        new_node.parents = parents

        # creating a new branch, and removing the old branches
        new_branch = Branch()
        new_branch.head = new_node
        self.branches = [new_branch]

    def append(self, element: Union[T, Feat, BaseProcessor]):
        if isinstance(element, (Feat, BaseProcessor)):
            if len(self.branches) == 0:
                self.create_branch(element)
            elif len(self.branches) == 1:
                self.branches[0].append(element)
            else:
                self.join_branches(element)
        elif isinstance(element, T):
            if len(self.branches) == 1:
                self.split_branch(element)
            else:
                self.add_tuple(element)
        else:
            raise PipelineBuildError(PIPELINE_TYPE_ERROR.format(obj_type=type(element)))


class ExtractionPipeline:
    """An extraction pipeline is a small DAG that describes the steps
    needed to extract one or several features."""

    def __init__(self,
                 elements: List[Union[BaseProcessor, T, Feat]],
                 fail_on_error: bool = True):
        self.elements = elements
        self.fail_on_error = fail_on_error

    def __rshift__(self, other: PipelineElement):
        if isinstance(other, (BaseProcessor, T, Feat)):
            self.elements.append(other)
        elif callable(other):
            self.elements.append(FunctionWrapperProcessor(other))
        elif isinstance(other, ExtractionPipeline):
            self.elements += other.elements
        else:
            raise PipelineBuildError(PIPELINE_TYPE_ERROR.format(obj_type=type(other)))

    def check(self):
        pass
        # TODO :
        #  * has to end with a feature or a feature tuple
        #  * has to start with inputs, input tuples or feature input
        #  *

    def build_pipeline_tree(self) -> PipelineDAG:
        pipeline_dag = PipelineDAG()
        for element in self.elements:
            pipeline_dag.append(element)
        return pipeline_dag
