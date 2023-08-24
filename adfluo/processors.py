import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass
from dis import get_instructions
from inspect import signature
from typing import Any, List, Tuple, Callable, TYPE_CHECKING, Hashable, Optional, Union, Set, Iterable, Dict

from sortedcontainers import SortedDict

from .dataset import Sample
from .exceptions import InvalidInputData
from .utils import logger, extraction_policy
from .validator import ValidatorFunction

if TYPE_CHECKING:
    from .pipeline import PipelineElement


@dataclass(frozen=True)
class ProcessorParameter:
    default: Hashable


# TODO : take in consideration hashing
@dataclass(frozen=True)
class ExtractorHyperParameter:
    name: str


def param(default: Optional[Hashable] = None) -> Any:
    return ProcessorParameter(default)


def hparam(name: str) -> Any:
    return ExtractorHyperParameter(name)


class ProcessorBase(ABC):
    """Abstract base class for a processor from the feature extraction pipeline"""

    def __init__(self, **kwargs):
        param_names = set(self.class_params)
        # setting kwargs-defined parameter values
        for key, val in kwargs.items():
            if key not in param_names:
                raise AttributeError(f"Attribute {key} isn't a processor parameter")
            try:
                hash(val)
            except TypeError:
                raise ValueError(f"Value for parameter {key} isn't hashable.")

            setattr(self, key, val)
            param_names.remove(key)

        # remaining parameters are set to the default set in the class attribute
        for param_key in param_names:
            proc_param: ProcessorParameter = getattr(self, param_key)
            setattr(self, param_key, proc_param.default)
        self._current_sample: Optional[Sample] = None

        self.post_init()

    def post_init(self):
        """To be overloaded by a child class, to do the usual job of the actual __init__ function"""
        pass

    @property
    def current_sample(self):
        return self._current_sample

    @property
    def nb_args(self):
        return len(signature(self.process).parameters)

    @property
    @abstractmethod
    def output_type(self):
        pass

    @abstractmethod
    def process(self, *args) -> Any:
        """Processes just one sample"""
        raise NotImplemented()

    @property
    def class_params(self) -> Set[str]:
        return {k for k, v in self.__class__.__dict__.items()
                if isinstance(v, ProcessorParameter)}

    @property
    def hparams(self) -> Set[str]:
        return {v.name for v in self.__dict__.values()
                if isinstance(v, ExtractorHyperParameter)}

    @property
    def _sorted_params(self) -> SortedDict:
        param_dict = SortedDict()
        for k in self.class_params:
            param_dict[k] = getattr(self, k, None)
        return param_dict

    def set_hparams(self, **params: Dict[str, Any]):
        for param in self.class_params:
            attribute_val = getattr(self, param, None)
            if not isinstance(attribute_val, ExtractorHyperParameter):
                continue
            # if the (class) processor parameter has been set as an hyperparam,
            # set it using values from the params dict
            if attribute_val.name in params:
                setattr(self, param, params[attribute_val.name])

    def __hash__(self):
        return hash((self.__class__, tuple(self._sorted_params.items())))

    def __eq__(self, other):
        return hash(self) == hash(other)

    def __repr__(self):
        return f"<{str(self)}>"

    def __str__(self):
        return "{class_name}({args})".format(
            class_name=self.__class__.__name__,
            args=",".join(f"{key}={value!r}" for key, value in self._sorted_params.items())
        )

    def __rshift__(self, other: 'PipelineElement'):
        from .pipeline import (ExtractionPipeline, PIPELINE_TYPE_ERROR,
                               PipelineBuildError)
        new_pipeline = ExtractionPipeline()
        new_pipeline.append(self)
        if isinstance(other, ProcessorBase):
            new_pipeline.append(other)
        elif isinstance(other, ExtractionPipeline):
            new_pipeline.concatenate(other)
        else:
            raise PipelineBuildError(PIPELINE_TYPE_ERROR.format(obj_type=type(other)))
        return new_pipeline

    def __or__(self, other: 'PipelineElement'):
        from .pipeline import (ExtractionPipeline, PIPELINE_TYPE_ERROR,
                               PipelineBuildError)
        new_pipeline = ExtractionPipeline()
        new_pipeline.append(self)
        if isinstance(other, ProcessorBase):
            new_pipeline.add_parallel_proc(other)
        elif isinstance(other, ExtractionPipeline):
            new_pipeline.add_parallel_pipeline(other)
        else:
            raise PipelineBuildError(PIPELINE_TYPE_ERROR.format(obj_type=type(other)))
        return new_pipeline


class SampleProcessor(ProcessorBase):
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
            print(extraction_policy.skip_errors)
            if extraction_policy.skip_errors:
                logger.warning("Got error in processor %s on sample %s : %s" %
                               (type(self).__name__, sample.id, str(e)))
                return None
            else:
                raise err
        else:
            return processed_sample_data

    @property
    def output_type(self):
        try:
            return self.process.__annotations__["return"]
        except KeyError:
            return Any


class FunctionWrapperMixin:
    """Mixin class for processors that wrap a function"""

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
        return hash((self.__class__, self.fun.__name__, instructions))

    @property
    def output_type(self):
        try:
            return self.fun.__annotations__["return"]
        except KeyError:
            return Any


class FunctionWrapperProcessor(FunctionWrapperMixin, SampleProcessor):
    """Used to wrap simple functions that can be used inline, without
    a processor"""

    def __repr__(self):
        return f"<{str(self)}>"

    def __str__(self):
        return f"F({self.fun.__name__})"

    def process(self, *args):
        return self.fun(*args)


F = FunctionWrapperProcessor
F.__doc__ = FunctionWrapperProcessor.__doc__


class ListWrapperProcessor(SampleProcessor):
    # TODO: double check and write some tests

    def __init__(self, proc: SampleProcessor):
        super().__init__()
        self.proc = proc

    def __repr__(self):
        return f"<{str(self)}>"

    def __str__(self):
        return f"L({str(self.proc)})"

    def __hash__(self):
        return hash((self.__class__, self.proc))

    @property
    def nb_args(self):
        return 1

    @property
    def output_type(self):
        try:
            return List[self.process.proc["return"]]
        except KeyError:
            return Any

    def process(self, arg: Iterable[Any]) -> List[Any]:
        return [self.proc(self._current_sample, (sub_sample,)) for sub_sample in arg]


L = ListWrapperProcessor
L.__doc__ = ListWrapperProcessor.__doc__


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
        self.validator_fn: Optional[ValidatorFunction] = None

    def process(self, *args) -> Any:
        data = self.current_sample[self.data_name]

        if self.validator_fn is not None:
            if not self.validator_fn(data):
                raise InvalidInputData(self.data_name, self.current_sample.id)
        return data

    def __str__(self):
        return f"Input({self.data_name})"


Input = SampleInputProcessor
Input.__doc__ = SampleInputProcessor.__doc__


class PrinterProcessor(SampleProcessor):
    name: str = param()

    def __init__(self, name: Optional[str] = None):
        super().__init__(name=name)

    def process(self, *args) -> Tuple:
        print(f"{self.name} received {args} ")
        return args


Printer = PrinterProcessor()


class SampleFeatureProcessor(SampleProcessor):
    """A passthrough processor used as a """
    feat_name: str = param()

    def __init__(self, feat_name: str):
        super().__init__(feat_name=feat_name)

    def process(self, *args) -> Tuple:
        return args[0]

    def __str__(self):
        return f"Feat({self.feat_name})"


Feat = SampleFeatureProcessor
Feat.__doc__ = SampleInputProcessor.__doc__


class DatasetAggregator(ProcessorBase):

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
