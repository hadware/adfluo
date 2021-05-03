from typing import Callable, Any

ValidatorFunction = Callable[[Any], bool]


class Validator:

    def __new__(cls, *args, **kwargs):
        validated_data_names = set()
        for name, method in cls.__dict__.items():
            try:
                data_name = method.validates_data_name
            except AttributeError:
                continue

            if data_name in validated_data_names:
                raise ValueError("There cannot be two validating functions for "
                                 "the same input data.")
            validated_data_names.add(data_name)
        return super().__new__(cls, *args, **kwargs)

    def get_validators(self):
        for name, method in self.__class__.__dict__.items():
            try:
                data_name = method.validates_data_name
            except AttributeError:
                continue

            # to ensure we're getting the bounded method, we're using getattr
            yield data_name, getattr(self, name)


def validates_input(data_name: str):
    def wrapper(method: ValidatorFunction):
        method.validates_data_name = data_name
        return method

    return wrapper
