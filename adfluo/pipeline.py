from typing import List, Union, Dict, Any

from .dataset import Sample, DictSample
from .exceptions import PipelineBuildError, PIPELINE_TYPE_ERROR
from .extraction_graph import SampleProcessorNode, BatchProcessorNode, FeatureNode, InputNode, FeatureName
from .processors import ProcessorBase, SampleProcessor, BatchProcessor, \
    Input, Feat

PipelineElement = Union['ExtractionPipeline', ProcessorBase]
ProcessorNode = Union[SampleProcessorNode, BatchProcessorNode]


def wrap_processor(proc: ProcessorBase) -> ProcessorNode:
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


# TODO : add "connect" function and streamline some of this code
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

    def check(self):
        """
        Checks that :
         - input nodes are all `InputNode` or `FeatureNode`
         - output nodes are all `FeatureNode`
         - "inner" nodes can't be input/feature nodes
         """

        # TODO : better error
        for node in self.inputs:
            assert isinstance(node, (InputNode, FeatureNode))
        for node in self.outputs:
            assert isinstance(node, FeatureNode)
        for node in self.all_nodes:
            if node not in self.inputs + self.outputs:
                assert not isinstance(node, (FeatureNode, InputNode))

    def append(self, proc: ProcessorBase):
        new_node = wrap_processor(proc)
        # extraction DAG has not node: new dag!
        if self.nb_inputs == 0:
            self.inputs = [new_node]

        # "regular" case: adding a node after the last one
        elif self.nb_outputs == 1:
            old_tail = self.outputs[0]
            old_tail.children = [new_node]
            new_node.parents = [old_tail]

        # adding a node as a merger of several branches
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
                i.parents = [self.outputs[0]]

        elif pipeline.nb_inputs == 1 and self.outputs > 1:
            # TODO: better error
            assert self.nb_outputs == pipeline.inputs[0].processor.nb_args
            pipeline.inputs[0].parents = self.outputs
            for o in self.outputs:
                o.children = [pipeline.inputs[0]]

        self.outputs = pipeline.outputs
        self.all_nodes += pipeline.all_nodes

    def merge_proc(self, proc: ProcessorBase):
        new_node = wrap_processor(proc)
        self.inputs.append(new_node)
        self.outputs.append(new_node)
        self.all_nodes.append(new_node)

    def merge_pipeline(self, pipeline: 'ExtractionPipeline'):
        self.inputs += pipeline.inputs
        self.outputs += pipeline.outputs
        self.all_nodes += pipeline.all_nodes

    def __rshift__(self, other: PipelineElement):
        if isinstance(other, ProcessorBase):
            self.append(other)
        elif isinstance(other, ExtractionPipeline):
            self.concatenate(other)
        else:
            raise PipelineBuildError(PIPELINE_TYPE_ERROR.format(obj_type=type(other)))
        return self

    def __add__(self, other: PipelineElement):
        if isinstance(other, ProcessorBase):
            self.merge_proc(other)
        elif isinstance(other, ExtractionPipeline):
            self.merge_pipeline(other)
        else:
            raise PipelineBuildError(PIPELINE_TYPE_ERROR.format(obj_type=type(other)))
        return self

    def __call__(self, sample: Union[Sample, Dict[str, Any]]) -> Dict[FeatureName, Any]:
        if isinstance(sample, dict):
            sample = DictSample(sample, 0)
        output_dict: Dict[FeatureName, Any] = {}
        # TODO: catch some errors (due to unsolved features/batch proc with no dataset)
        #  and "contextualize" them
        for output_node in self.outputs:
            output_node: FeatureNode
            output_dict[output_node.processor.feat_name] = output_node[sample]
        return output_dict
