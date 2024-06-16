from abc import abstractmethod, ABCMeta
from dataclasses import dataclass
from dis import get_instructions
from inspect import signature
from typing import Any, Callable, TYPE_CHECKING, Hashable, Optional, Iterable, Type

from sortedcontainers import SortedDict

from . import DatasetLoader
from .dataset import Sample
from .exceptions import InvalidInputData
from .storage import StorageProtocol
from .types import SampleID
from .utils import logger
from .validator import ValidatorFunction

if TYPE_CHECKING:
    from .pipeline import PipelineElement


@dataclass(frozen=True)
class ParametersCount:
    nb_args: int
    variable: bool

    def accept(self, nb_args: int):
        return nb_args >= self.nb_args if self.variable else nb_args == self.nb_args


@dataclass(frozen=True)
class ProcessorParameter:
    default: Hashable


# TODO : take in consideration hashing
@dataclass(frozen=True)
class ExtractorHyperParameter:
    name: str
    type: Optional[Type]


def param(default: Optional[Hashable] = None) -> Any:
    return ProcessorParameter(default)


def hparam(name: str, type: Optional[Type] = None) -> Any:
    return ExtractorHyperParameter(name, type)


class ProcessorBase(metaclass=ABCMeta):
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
                raise ValueError(f"Value for parameter {key} isn't hashable and has to be.")

            setattr(self, key, val)
            param_names.remove(key)

        # store hparams.name -> (hparam, attr name) in a dict that shouldn't be changed when hparams are set
        self._hparams = {hparam.name: (hparam, attr_name)
                         for attr_name, hparam in self.__dict__.items()
                         if isinstance(hparam, ExtractorHyperParameter)}

        # remaining parameters are set to the default set in the class attribute
        for param_key in param_names:
            proc_param: ProcessorParameter = getattr(self, param_key)
            setattr(self, param_key, proc_param.default)
        self._current_sample: Optional[Sample] = None

        if len(self.signature.parameters) < 1:
            raise ValueError("Function must have at least one parameter")

        self.post_init()

    def post_init(self):
        """To be overloaded by a child class, to do the usual job of the actual __init__ function"""
        pass

    @property
    def current_sample(self):
        return self._current_sample

    @property
    @abstractmethod
    def signature(self):
        pass

    @property
    def parameters(self):
        # 2 is "VAR_POSITIONAL"
        variable_params = any(param.kind == 2 for param in self.signature.parameters.values())
        return ParametersCount(len(self.signature.parameters), variable_params)

    @property
    @abstractmethod
    def output_type(self):
        pass

    @property
    def class_params(self) -> set[str]:
        # TODO: maybe use dir(self.__class__) to allow for inheritance
        return {k for k, v in self.__class__.__dict__.items()
                if isinstance(v, ProcessorParameter)}

    @property
    def hparams(self) -> set[str]:
        return set(self._hparams.keys())

    @property
    def unset_hparams(self) -> set[str]:
        return {hparam.name for hparam in self.__dict__.values()
                if isinstance(hparam, ExtractorHyperParameter)}

    @property
    def _sorted_params(self) -> SortedDict:
        param_dict = SortedDict()
        for k in self.class_params:
            param_dict[k] = getattr(self, k, None)
        return param_dict

    def set_hparams(self, **hparams: dict[str, Any]):
        # setting only the hparams that are injected in this processors
        for hparam_name in set(self._hparams.keys()) | self.hparams:
            hparam, hparam_attr = self._hparams[hparam_name]
            proc_param_value = hparams[hparam_name]
            # converting param value using hparam type if specified
            if hparam.type is not None:
                proc_param_value = hparam.type(proc_param_value)
            setattr(self, hparam_attr, proc_param_value)

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


class SampleProcessor(ProcessorBase, metaclass=ABCMeta):
    """Processes one sample after the other, independently"""

    @property
    def signature(self):
        return signature(self.process)

    @abstractmethod
    def process(self, *args) -> Any:
        """Processes just one sample"""
        raise NotImplemented()

    @property
    def output_type(self):
        try:
            return self.process.__annotations__["return"]
        except KeyError:
            return Any

    def __call__(self, sample: Sample, sample_data: tuple[Any]) -> Any:
        self._current_sample = sample
        return self.process(*sample_data)


class FunctionWrapperMixin:
    """Mixin class for processors that wrap a function"""

    def __init__(self, fun: Callable):
        self.fun = fun
        super().__init__()

    @property
    def signature(self):
        return signature(self.fun)

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
    """Akin to a "map" function, applied on a list of per-sample data"""

    # TODO: double check and write some tests
    # TODO: add support for datasetprocessors? (via a ListWrapperMixin?, c.f. FunctionWrapper)

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
    def output_type(self):
        try:
            return list[self.process.proc["return"]]
        except KeyError:
            return Any

    def process(self, arg: Iterable[Any]) -> list[Any]:
        return [self.proc(self._current_sample, (sub_sample,)) for sub_sample in arg]


L = ListWrapperProcessor
L.__doc__ = ListWrapperProcessor.__doc__


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


class DatasetInputProcessor(SampleProcessor):
    """Processor that pulls data from a dataset."""
    data_name: str = param()

    def __init__(self, data_name: str):
        super().__init__(data_name=data_name)
        self.validator_fn: Optional[ValidatorFunction] = None
        self.dataset: DatasetLoader = None

    def process(self, *args) -> Any:
        data = self.dataset[self.data_name]

        if self.validator_fn is not None:
            if not self.validator_fn(data):
                raise InvalidInputData(self.data_name, self.current_sample.id)
        return data

    def __str__(self):
        return f"DSInput({self.data_name})"


DSInput = DatasetInputProcessor
DSInput.__doc__ = DatasetInputProcessor.__doc__


class LoggerProcessor(SampleProcessor):
    name: str = param()

    def __init__(self, name: Optional[str] = None, formatter: Optional[Callable[[Any], Any]] = None):
        super().__init__(name=name)
        self.formatter = formatter

    def process(self, *args) -> tuple:
        msg = str(self.formatter(*args)) if self.formatter is not None else str(args)
        if self.name is not None:
            msg = f"{self.name}: {msg}"
        logger.info(msg)
        return args


Logger = LoggerProcessor
Pass = F(lambda x: x)


class BaseFeat(SampleProcessor, metaclass=ABCMeta):
    """Base class for Features and Dataset Features"""
    feat_name: str

    def __init__(self, feat_name: str,
                 storage: Optional[StorageProtocol] = None,
                 default: Any | Callable[[], Any] = None):
        super().__init__(feat_name=feat_name)
        self.custom_storage = storage
        self.default = default

    def process(self, *args) -> tuple:
        return args[0]


class SampleFeatureProcessor(BaseFeat):
    """A passthrough processor used to indicate per-sample features (Feats)"""
    feat_name: str = param()

    # TODO: doc
    def __str__(self):
        return f"Feat({self.feat_name})"


class DatasetFeatureProcessor(BaseFeat):
    """A passthrough processor used to indicate Dataset Features (DSFeats)"""
    feat_name: str = param()

    # TODO: doc
    def __str__(self):
        return f"DSFeat({self.feat_name})"


Feat = SampleFeatureProcessor
Feat.__doc__ = SampleInputProcessor.__doc__

DSFeat = DatasetFeatureProcessor
DSFeat.__doc__ = DatasetFeatureProcessor.__doc__


class DatasetAggregator(ProcessorBase, metaclass=ABCMeta):

    @property
    def signature(self):
        return signature(self.aggregate)

    @property
    def output_type(self):
        try:
            return self.aggregate.__annotations__["return"]
        except KeyError:
            return Any

    @abstractmethod
    def aggregate(self, samples_data: list[Any] | list[tuple[Any, ...]]) -> Any:
        pass

    def __call__(self, samples_data: dict[SampleID, tuple[Any, ...]]):
        # Call aggregate
        first_value = next(iter(samples_data.values()))
        if len(first_value) == 1:
            return self.aggregate([t[0] for t in samples_data.values()])
        else:
            return self.aggregate(list(samples_data.values()))


class FunctionWrapperAggregator(FunctionWrapperMixin, DatasetAggregator):

    def __repr__(self):
        return f"Aggregator({self.fun.__name__})"

    def aggregate(self, samples_data: list[Any | tuple]) -> Any:
        return self.fun(samples_data)


Agg = FunctionWrapperAggregator
Agg.__doc__ = FunctionWrapperProcessor.__doc__
