import sys
from abc import ABC, abstractmethod
from dis import get_instructions
from inspect import signature
from typing import Any, List, Tuple, Callable, TYPE_CHECKING, Hashable, Optional, Union, Set

from sortedcontainers import SortedDict

from .dataset import Sample
from .utils import logger, extraction_policy

if TYPE_CHECKING:
    from .pipeline import PipelineElement


class ProcessorParameter:
    def __init__(self, value: Hashable):
        self.value: Hashable = value


def param(default: Optional[Hashable] = None) -> Any:
    return ProcessorParameter(default)


class BaseProcessor(ABC):
    """Base class for a processor from the feature extraction pipeline"""

    def __init__(self, **kwargs):
        param_names = set(k for k, v in self.__class__.__dict__.items()
                          if isinstance(v, ProcessorParameter))
        # setting kwargs-defined parameter values
        for key, val in kwargs.items():
            if key not in param_names:
                raise AttributeError(f"Attribute {key} isn't a processor parameter")
            setattr(self, key, val)
            param_names.remove(key)

        # remaining parameters are set to the default set in the class attribute
        for param_key in param_names:
            proc_param: ProcessorParameter = getattr(self, param_key)
            setattr(self, param_key, proc_param.value)
        self._current_sample: Optional[Sample] = None

        self.post_init()

    def post_init(self):
        pass

    @property
    def current_sample(self):
        return self._current_sample

    @property
    def nb_args(self):
        return len(signature(self.process).parameters)

    @abstractmethod
    def process(self, *args) -> Any:
        """Processes just one sample"""
        raise NotImplemented()

    @property
    def _class_params(self) -> Set[str]:
        return {k for k, v in self.__class__.__dict__.items()
                if isinstance(v, ProcessorParameter)}

    @property
    def _params(self) -> SortedDict:
        class_params = self._class_params
        param_dict = SortedDict()
        for k, v in self.__class__.__dict__.items():
            if k in class_params:
                param_dict[k] = getattr(self, k, None)
        return param_dict

    def __hash__(self):
        return hash((self.__class__, tuple(self._params.items())))

    def __eq__(self, other):
        return hash(self) == hash(other)

    def __repr__(self):
        return "<%s(%s)>" % (
            self.__class__.__name__,
            ",".join(f"{key}={value!r}" for key, value in self._params.items())
        )

    def __rshift__(self, other: 'PipelineElement'):
        from .pipeline import (ExtractionPipeline, PIPELINE_TYPE_ERROR,
                               PipelineBuildError)
        new_pipeline = ExtractionPipeline()
        new_pipeline.append(self)
        if isinstance(other, BaseProcessor):
            new_pipeline.append(other)
        elif isinstance(other, ExtractionPipeline):
            new_pipeline.concatenate(other)
        else:
            raise PipelineBuildError(PIPELINE_TYPE_ERROR.format(obj_type=type(other)))
        return new_pipeline

    def __add__(self, other: 'PipelineElement'):
        from .pipeline import (ExtractionPipeline, PIPELINE_TYPE_ERROR,
                               PipelineBuildError)
        new_pipeline = ExtractionPipeline()
        new_pipeline.append(self)
        if isinstance(other, BaseProcessor):
            new_pipeline.merge_proc(other)
        elif isinstance(other, ExtractionPipeline):
            new_pipeline.merge_pipeline(other)
        else:
            raise PipelineBuildError(PIPELINE_TYPE_ERROR.format(obj_type=type(other)))
        return new_pipeline


class SampleProcessor(BaseProcessor):
    """Processes one sample after the other, independently"""

    def __call__(self, sample: Sample, sample_data: Tuple[Any]) -> Any:

        # trying to process the sample. If an error is raised, two
        # possible outcomes:
        # - if we have to fail on error, than the error is raised
        # (with a cleaner call stack)
        # - if not, we just replace the processed output with None
        # and print an error message
        try:
            self._current_sample = sample
            processed_sample_data = self.process(*sample_data)
        except Exception as e:
            tb = sys.exc_info()[2]
            err = type(e)(("In processor %s, on sample %s : "
                           % (repr(self), sample.id)) +
                          str(e)).with_traceback(tb)
            if extraction_policy.skip_errors:
                logger.warning("Got error in processor %s on sample %s : %s" %
                               (type(self).__name__, sample.id, str(e)))
                return None
            else:
                raise err
        else:
            return processed_sample_data


class FunctionWrapperMixin:

    def __init__(self, fun: Callable):
        self.fun = fun
        if len(signature(fun).parameters) < 1:
            raise ValueError("Function must have at least one parameter")

    @property
    def nb_args(self):
        return len(signature(self.fun).parameters)

    def __hash__(self):
        """Hashes the disassembled code of the wrapped function."""
        instructions = tuple((instr.opname, instr.arg, instr.argval)
                             for instr in get_instructions(self.fun))
        return hash(instructions)


class FunctionWrapperProcessor(FunctionWrapperMixin, SampleProcessor):
    """Used to wrap simple functions that can be used inline, without
    a processor"""

    def __repr__(self):
        return f"{self.fun.__name__}"

    def process(self, *args):
        return self.fun(*args)


F = FunctionWrapperProcessor


class BatchProcessor(SampleProcessor):
    """Processor class that requires the full list of samples from the
    dataset to be able to process individual samples"""

    @abstractmethod
    def full_dataset_process(self, samples_data: List[Union[Any, Tuple]]):
        """Processes the full dataset of samples. Doesn't return anything,
        store the results as instance attributes"""
        pass


class SampleInputProcessor(SampleProcessor):
    """Processor that pulls data from samples."""
    data_name: str = param()

    def __init__(self, data_name: str):
        super().__init__(data_name=data_name)

    def process(self, *args) -> Any:
        return self.current_sample[self.data_name]


Input = SampleInputProcessor


class PrinterProcessor(SampleProcessor):
    name: str = param()

    def __init__(self, name: Optional[str] = None):
        super().__init__(name=name)

    def process(self, *args) -> Tuple:
        print(f"{self.name} received {args} ")
        return args


Printer = PrinterProcessor()


class SampleFeatureProcessor(SampleProcessor):
    """A subtype of `PassThroughProcessor` used to reference a feature"""
    feat_name: str = param()

    def __init__(self, feat_name: str):
        super().__init__(feat_name=feat_name)

    def process(self, *args) -> Tuple:
        return args[0]


Feat = SampleFeatureProcessor


class DatasetAggregator(BaseProcessor):

    @abstractmethod
    def aggregate(self, samples_data: List[Union[Any, Tuple]]) -> Any:
        pass

    def process(self, *args) -> Any:
        pass

    def __call__(self, *args, **kwargs):
        pass  # TODO


class FunctionWrapperAggregator(DatasetAggregator, FunctionWrapperMixin):

    def __repr__(self):
        return f"Aggregator({self.fun.__name__})"

    def aggregate(self, samples_data: List[Union[Any, Tuple]]) -> Any:
        return self.fun(samples_data)


Aggregator = FunctionWrapperAggregator
