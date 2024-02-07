from dataclasses import dataclass
from types import TracebackType
from typing import Optional, TYPE_CHECKING

from adfluo import Sample
from adfluo.types import SampleID

if TYPE_CHECKING:
    from adfluo.processors import SampleProcessor

PIPELINE_TYPE_ERROR = "Invalid object in pipeline of type {obj_type}"


class PipelineBuildError(Exception):
    pass


class BadSampleException(RuntimeError):

    def __init__(self, sample: Sample, *args):
        self.sample = sample
        super().__init__(*args)


@dataclass
class ExtractionContext:
    sample: SampleID
    processor: 'SampleProcessor'
    feature: str = None


class ExtractionError(RuntimeError):

    def __init__(self, error: Exception,
                 ctx: ExtractionContext, *args):
        super().__init__(*args)
        self.error = error
        self.ctx = ctx

    def __str__(self):
        return (f"In processor: {str(self.ctx.processor)}, "
                f"on sample {self.ctx.sample}, "
                f"when computing feature {self.ctx.feature}, "
                f"{str(type(self.error))} : {str(self.error)}")


class BadAggregationException(RuntimeError):
    pass


class UnsolvedFeatureDependencyError(RuntimeError):
    pass


class DuplicateSampleError(ValueError):

    def __init__(self, sample_id: str, *args):
        super().__init__(f"Two samples share the same id '{sample_id}'",
                         *args)


class InvalidInputData(ValueError):

    def __init__(self, data_name: str, sample_id: str, *args):
        super().__init__(f"Input data '{data_name}' in sample {sample_id} "
                         f"is invalid.", *args)
