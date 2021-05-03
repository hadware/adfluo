from adfluo import Sample

PIPELINE_TYPE_ERROR = "Invalid object in pipeline of type {obj_type}"


class PipelineBuildError(Exception):
    pass


class BadSampleException(RuntimeError):

    def __init__(self, sample: Sample, *args):
        self.sample = sample
        super().__init__(*args)


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
