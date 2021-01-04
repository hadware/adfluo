from typing import List, Union

from .processors import BaseProcessor

PipelineElement = Union['ExtractionPipeline', BaseProcessor, 'T', 'Feat']

PIPELINE_TYPE_ERROR = "Invalid object in pipeline of type {obj_type}"


class Feat:

    def __init__(self, feat_name: str):
        self.feat_name = feat_name


class T:
    def __init__(self, *args):
        for arg in args:
            if not isinstance(arg, (Feat, BaseProcessor)):
                raise TypeError(PIPELINE_TYPE_ERROR.format(obj_type=type(args)))
        self.procs = args

    def __gt__(self, other: PipelineElement):
        if isinstance(other, (BaseProcessor, T)):
            return ExtractionPipeline([self, other])
        elif isinstance(other, ExtractionPipeline):
            other.pipeline.insert(0, self)
        else:
            raise TypeError(PIPELINE_TYPE_ERROR.format(obj_type=type(other)))


class ExtractionPipeline:
    def __init__(self,
                 processors: List[Union[BaseProcessor, T]],
                 fail_on_error: bool = True):
        self.pipeline = processors
        self.fail_on_error = fail_on_error

    def __gt__(self, other: PipelineElement):
        if isinstance(other, (BaseProcessor, T)):
            self.pipeline.append(other)
        elif isinstance(other, ExtractionPipeline):
            self.pipeline += other.pipeline
        else:
            raise TypeError(PIPELINE_TYPE_ERROR.format(obj_type=type(other)))

    def check(self):
        pass
    # TODO :
    #  * has to end with a feature or a feature tuple
    #  * has to start with inputs, input tuples or feature input
    #  *
