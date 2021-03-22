from collections import deque
from typing import List, Union, Deque

from .extraction_graph import BaseGraphNode, SampleProcessorNode, BatchProcessorNode, FeatureNode, InputNode
from .processors import BaseProcessor, FunctionWrapperProcessor, SampleProcessor, BatchProcessor, \
    Input, Feat

PipelineElement = Union['ExtractionPipeline', BaseProcessor, 'T', 'Feat']
ProcessorNode = Union[SampleProcessorNode, BatchProcessorNode]

PIPELINE_TYPE_ERROR = "Invalid object in pipeline of type {obj_type}"


class PipelineBuildError(Exception):
    pass


def wrap_processor(proc: BaseProcessor) -> ProcessorNode:
    if isinstance(proc, Feat):
        return FeatureNode(proc)
    elif isinstance(proc, Input):
        return InputNode(proc)
    elif isinstance(proc, BatchProcessor):
        return BatchProcessorNode(proc)
    elif isinstance(proc, SampleProcessor):
        return SampleProcessorNode(proc)
    else:
        raise PipelineBuildError(PIPELINE_TYPE_ERROR.format(obj_type=type(proc)))

# TODO : add "connect" function and steamline some of this code
class ExtractionPipeline:

    def __init__(self):
        self.inputs: List[ProcessorNode] = []
        self.outputs: List[ProcessorNode] = []
        self.all_nodes: List[ProcessorNode] = []

    @property
    def nb_inputs(self):
        return len(self.inputs)

    @property
    def nb_outputs(self):
        return len(self.outputs)

    def walk(self):
        # TODO : rework this.
        #  - Input/output nodes still need to be checked
        #  - "inner" nodes can't be input/feature nodes

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

    def append(self, proc: BaseProcessor):
        new_node = wrap_processor(proc)
        if self.nb_inputs == 0:
            self.inputs = [new_node]
        elif self.nb_outputs == 1:
            old_tail = self.inputs[-1]
            old_tail.children = [new_node]
            new_node.parents = [old_tail]
        else:
            assert self.nb_outputs == proc.nb_args
            new_node.parents = self.outputs
            for o in self.outputs:
                o.children = [new_node]
        self.outputs = [new_node]

        self.all_nodes.append(new_node)

    def concatenate(self, pipeline: 'ExtractionPipeline'):
        if self.nb_outputs == pipeline.nb_inputs:
            for right_out, left_in in zip(self.outputs, pipeline.inputs):
                right_out.children = [left_in]
                left_in.parents = [right_out]

        elif self.nb_outputs == 1 and pipeline.nb_inputs > 1:
            self.outputs[0].children = pipeline.inputs
            for i in pipeline.inputs:
                i.parents = self.outputs[0]

        elif pipeline.nb_inputs == 1 and self.outputs > 1:
            # TODO: better error
            assert self.nb_outputs == pipeline.inputs[0].processor.nb_args
            pipeline.inputs[0].parents = self.outputs
            for o in self.outputs:
                o.children = pipeline.inputs[0]

        self.outputs = pipeline.outputs
        self.all_nodes += pipeline.all_nodes

    def merge_proc(self, proc: BaseProcessor):
        new_node = wrap_processor(proc)
        self.inputs.append(new_node)
        self.outputs.append(new_node)
        self.all_nodes.append(new_node)

    def merge_pipeline(self, pipeline: 'ExtractionPipeline'):
        self.inputs += pipeline.inputs
        self.outputs += pipeline.outputs
        self.all_nodes += pipeline.all_nodes

    def __rshift__(self, other: PipelineElement):
        if isinstance(other, BaseProcessor):
            self.append(other)
        elif isinstance(other, ExtractionPipeline):
            self.concatenate(other)
        elif callable(other):
            self.append(FunctionWrapperProcessor(other))
        else:
            raise PipelineBuildError(PIPELINE_TYPE_ERROR.format(obj_type=type(other)))
        return self

    def __add__(self, other: PipelineElement):
        if isinstance(other, BaseProcessor):
            self.merge_proc(other)
        elif isinstance(other, ExtractionPipeline):
            self.merge_pipeline(other)
        elif callable(other):
            self.merge_proc(FunctionWrapperProcessor(other))
        else:
            raise PipelineBuildError(PIPELINE_TYPE_ERROR.format(obj_type=type(other)))
        return self
