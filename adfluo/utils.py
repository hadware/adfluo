import logging
import sys
import typing
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger("adfluo")


@dataclass
class ExtractionPolicy:
    skip_errors: bool = False
    no_cache: bool = False


NoneType = type(None)


# the following code is shamelessly copied from sphinx's codebase, and slightly modified

def format_annotation(annotation: Any) -> str:
    """Return formatted representation of a type annotation.
    Show qualified names for types and additional details for types from
    the ``typing`` module.
    Displaying complex types from ``typing`` relies on its private API.
    """
    if isinstance(annotation, str):
        return annotation
    elif isinstance(annotation, typing.TypeVar):
        return annotation.__name__
    elif not annotation:
        return repr(annotation)
    elif annotation is NoneType:
        return 'None'
    elif getattr(annotation, '__module__', None) == 'builtins':
        return annotation.__qualname__
    elif annotation is Ellipsis:
        return '...'

    if sys.version_info >= (3, 7):
        return format_annotation_new(annotation)
    else:
        return format_annotation_old(annotation)


def format_annotation_new(annotation: Any) -> str:
    """format_annotation() for py37+"""
    module = getattr(annotation, '__module__', None)
    if module == 'typing':
        if getattr(annotation, '_name', None):
            qualname = annotation._name
        elif getattr(annotation, '__qualname__', None):
            qualname = annotation.__qualname__
        elif getattr(annotation, '__forward_arg__', None):
            qualname = annotation.__forward_arg__
        else:
            qualname = format_annotation(annotation.__origin__)
    elif hasattr(annotation, '__qualname__'):
        qualname = annotation.__qualname__
    else:
        qualname = repr(annotation)

    if getattr(annotation, '__args__', None):
        if qualname == 'Union':
            if len(annotation.__args__) == 2 and annotation.__args__[1] is NoneType:
                return 'Optional[%s]' % format_annotation(annotation.__args__[0])
            else:
                args = ', '.join(format_annotation(a) for a in annotation.__args__)
                return '%s[%s]' % (qualname, args)
        elif qualname == 'Callable':
            args = ', '.join(format_annotation(a) for a in annotation.__args__[:-1])
            returns = format_annotation(annotation.__args__[-1])
            return '%s[[%s], %s]' % (qualname, args, returns)
        elif annotation._special:
            return qualname
        else:
            args = ', '.join(format_annotation(a) for a in annotation.__args__)
            return '%s[%s]' % (qualname, args)

    return qualname


def format_annotation_old(annotation: Any) -> str:
    """format_annotation() for py36 or below"""
    module = getattr(annotation, '__module__', None)
    if module == 'typing':
        if getattr(annotation, '_name', None):
            qualname = annotation._name
        elif getattr(annotation, '__qualname__', None):
            qualname = annotation.__qualname__
        elif getattr(annotation, '__forward_arg__', None):
            qualname = annotation.__forward_arg__
        elif getattr(annotation, '__origin__', None):
            qualname = format_annotation(annotation.__origin__)
        else:
            qualname = repr(annotation).replace('typing.', '')
    elif hasattr(annotation, '__qualname__'):
        qualname = annotation.__qualname__
    else:
        qualname = repr(annotation)

    if (isinstance(annotation, typing.TupleMeta) and
            not hasattr(annotation, '__tuple_params__')):
        params = annotation.__args__
        if params:
            param_str = ', '.join(format_annotation(p) for p in params)
            return '%s[%s]' % (qualname, param_str)
        else:
            return qualname
    elif isinstance(annotation, typing.GenericMeta):
        params = None
        if hasattr(annotation, '__args__'):

            if annotation.__args__ is None or len(annotation.__args__) <= 2:
                params = annotation.__args__
            else:
                args = ', '.join(format_annotation(arg) for arg
                                 in annotation.__args__[:-1])
                result = format_annotation(annotation.__args__[-1])
                return '%s[[%s], %s]' % (qualname, args, result)
        elif hasattr(annotation, '__parameters__'):

            params = annotation.__parameters__
        if params is not None:
            param_str = ', '.join(format_annotation(p) for p in params)
            return '%s[%s]' % (qualname, param_str)
    elif (hasattr(typing, 'UnionMeta') and
          isinstance(annotation, typing.UnionMeta) and
          hasattr(annotation, '__union_params__')):
        params = annotation.__union_params__
        if params is not None:
            if len(params) == 2 and params[1] is NoneType:
                return 'Optional[%s]' % format_annotation(params[0])
            else:
                param_str = ', '.join(format_annotation(p) for p in params)
                return '%s[%s]' % (qualname, param_str)
    elif (hasattr(annotation, '__origin__') and
          annotation.__origin__ is typing.Union):
        params = annotation.__args__
        if params is not None:
            if len(params) == 2 and params[1] is NoneType:
                return 'Optional[%s]' % format_annotation(params[0])
            else:
                param_str = ', '.join(format_annotation(p) for p in params)
                return 'Union[%s]' % param_str
    elif (isinstance(annotation, typing.CallableMeta) and
          getattr(annotation, '__args__', None) is not None and
          hasattr(annotation, '__result__')):

        args = annotation.__args__
        if args is None:
            return qualname
        elif args is Ellipsis:
            args_str = '...'
        else:
            formatted_args = (format_annotation(a) for a in args)
            args_str = '[%s]' % ', '.join(formatted_args)
        return '%s[%s, %s]' % (qualname,
                               args_str,
                               format_annotation(annotation.__result__))
    elif (isinstance(annotation, typing.TupleMeta) and
          hasattr(annotation, '__tuple_params__') and
          hasattr(annotation, '__tuple_use_ellipsis__')):
        params = annotation.__tuple_params__
        if params is not None:
            param_strings = [format_annotation(p) for p in params]
            if annotation.__tuple_use_ellipsis__:
                param_strings.append('...')
            return '%s[%s]' % (qualname,
                               ', '.join(param_strings))

    return qualname
