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
    def append_right(self, element: BaseProcessor):
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

    def append_right(self, element: BaseProcessor):
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
        self.input_nodes: Optional[List[SampleProcessorNode]] = None
        self.output_nodes: Optional[List[FeatureNode]] = None
        self.all_nodes: Optional[List[BaseGraphNode]] = None

    @property
    def head(self):
        assert len(self.branches) == 1
        return self.branches[0].head

    def walk(self):
        """Starting from the head of its branches, goes up the DAG to gather
        input and output nodes"""
        self.input_nodes, self.output_nodes, self.all_nodes = [], [], []
        nodes_stack: Deque[BaseGraphNode] = deque()
        for branch in self.branches:
            self.output_nodes.append(branch.head)
            # TODO: better error
            assert isinstance(branch.head, FeatureNode)
            nodes_stack.appendleft(branch.head)

        self.all_nodes += self.output_nodes

        while nodes_stack:
            # if a node on the stack has parents, put them on the stack, else,
            # it must be an input node, and put it in the input nodes list
            node = nodes_stack.pop()
            self.all_nodes.append(node)
            if node.parents:
                for parent in node.parents:
                    nodes_stack.appendleft(parent)
            else:
                self.input_nodes.append(node)

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
                new_pipeline.append_right(element)
                self.branches.append(new_pipeline)
            else:
                # TODO : better error
                raise PipelineBuildError(PIPELINE_TYPE_ERROR.format(obj_type=type(element)))

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

    def append_right(self, element: Union['PipelineDAG', BaseProcessor]):
        if isinstance(element, (Feat, BaseProcessor)):
            if len(self.branches) == 0:
                self.create_branch(element)
            elif len(self.branches) == 1:
                self.branches[0].append_right(element)
            else:
                self.join_branches(element)
        elif isinstance(element, T):
            if len(self.branches) == 1:
                self.split_branch(element)
            else:
                self.add_tuple(element)
        else:
            raise PipelineBuildError(PIPELINE_TYPE_ERROR.format(obj_type=type(element)))

    def append_left(self):
        pass


class ExtractionPipeline:
    """An extraction pipeline is a small DAG that describes the steps
    needed to extract one or several features."""

    def __init__(self,
                 elements: List[BaseProcessor]):
        self.pipeline_dag = PipelineDAG()
        for e in elements:
            self.pipeline_dag.append_right(e)

    def __rshift__(self, other: PipelineElement):
        if isinstance(other, (BaseProcessor, ExtractionPipeline)):
            self.pipeline_dag.append_right(other)
        elif callable(other):
            self.pipeline_dag.append_right(FunctionWrapperProcessor(other))
        else:
            raise PipelineBuildError(PIPELINE_TYPE_ERROR.format(obj_type=type(other)))

    def __add__(self, other: PipelineElement):
        # TODO
        if isinstance(other, BaseProcessor):
            self.elements.append_right(other)
        elif callable(other):
            self.elements.append_right(FunctionWrapperProcessor(other))
        elif isinstance(other, ExtractionPipeline):
            self.elements += other.elements
        else:
            raise PipelineBuildError(PIPELINE_TYPE_ERROR.format(obj_type=type(other)))

