"""
    This module contains functions and primitives extracted from the Sphinx project.
    They are used to get the short version of a function's return annotation.

    This code is thus under the following license :

        Copyright (c) 2007-2023 by the Sphinx team (see https://github.com/sphinx-doc/sphinx/blob/master/AUTHORS.rst file).
        All rights reserved.

        Redistribution and use in source and binary forms, with or without modification, are permitted provided that the
        following conditions are met:

         * Redistributions of source code must retain the above copyright notice,
           this list of conditions and the following disclaimer.
         * Redistributions in binary form must reproduce the above copyright notice, this list of conditions
         and the following disclaimer in the documentation and/or other materials provided with the distribution.

        THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
        INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
        ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
        INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
        PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
        HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
        OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
        EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""

import enum
import sys
import types
import typing
from contextvars import Context, ContextVar, Token
from struct import Struct
from typing import Any, TypeVar, Dict, Optional

if sys.version_info >= (3, 10):
    from types import UnionType
else:
    UnionType = None

# classes that have an incorrect .__module__ attribute
_INVALID_BUILTIN_CLASSES = {
    Context: 'contextvars.Context',  # Context.__module__ == '_contextvars'
    ContextVar: 'contextvars.ContextVar',  # ContextVar.__module__ == '_contextvars'
    Token: 'contextvars.Token',  # Token.__module__ == '_contextvars'
    Struct: 'struct.Struct',  # Struct.__module__ == '_struct'
    # types in 'types' with <type>.__module__ == 'builtins':
    types.AsyncGeneratorType: 'types.AsyncGeneratorType',
    types.BuiltinFunctionType: 'types.BuiltinFunctionType',
    types.BuiltinMethodType: 'types.BuiltinMethodType',
    types.CellType: 'types.CellType',
    types.ClassMethodDescriptorType: 'types.ClassMethodDescriptorType',
    types.CodeType: 'types.CodeType',
    types.CoroutineType: 'types.CoroutineType',
    types.FrameType: 'types.FrameType',
    types.FunctionType: 'types.FunctionType',
    types.GeneratorType: 'types.GeneratorType',
    types.GetSetDescriptorType: 'types.GetSetDescriptorType',
    types.LambdaType: 'types.LambdaType',
    types.MappingProxyType: 'types.MappingProxyType',
    types.MemberDescriptorType: 'types.MemberDescriptorType',
    types.MethodDescriptorType: 'types.MethodDescriptorType',
    types.MethodType: 'types.MethodType',
    types.MethodWrapperType: 'types.MethodWrapperType',
    types.ModuleType: 'types.ModuleType',
    types.TracebackType: 'types.TracebackType',
    types.WrapperDescriptorType: 'types.WrapperDescriptorType',
}


def is_invalid_builtin_class(obj: Any) -> bool:
    """Check *obj* is an invalid built-in class."""
    try:
        return obj in _INVALID_BUILTIN_CLASSES
    except TypeError:  # unhashable type
        return False


# type of None
NoneType = type(None)


def safe_getattr(obj: Any, name: str, *defargs: Any) -> Any:
    """A getattr() that turns all exceptions into AttributeErrors."""
    try:
        return getattr(obj, name, *defargs)
    except Exception as exc:
        # sometimes accessing a property raises an exception (e.g.
        # NotImplementedError), so let's try to read the attribute directly
        try:
            # In case the object does weird things with attribute access
            # such that accessing `obj.__dict__` may raise an exception
            return obj.__dict__[name]
        except Exception:
            pass

        # this is a catch-all for all the weird things that some modules do
        # with attribute access
        if defargs:
            return defargs[0]

        raise AttributeError(name) from exc


def get_type_hints(
        obj: Any, globalns: Optional[Dict[str, Any]] = None, localns: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Return a dictionary containing type hints for a function, method, module or class
    object.

    This is a simple wrapper of `typing.get_type_hints()` that does not raise an error on
    runtime.
    """

    try:
        return typing.get_type_hints(obj, globalns, localns)
    except NameError:
        # Failed to evaluate ForwardRef (maybe TYPE_CHECKING)
        return safe_getattr(obj, '__annotations__', {})
    except AttributeError:
        # Failed to evaluate ForwardRef (maybe not runtime checkable)
        return safe_getattr(obj, '__annotations__', {})
    except TypeError:
        # Invalid object is given. But try to get __annotations__ as a fallback.
        return safe_getattr(obj, '__annotations__', {})
    except KeyError:
        # a broken class found (refs: https://github.com/sphinx-doc/sphinx/issues/8084)
        return {}


def is_system_TypeVar(typ: Any) -> bool:
    """Check *typ* is system defined TypeVar."""
    modname = getattr(typ, '__module__', '')
    return modname == 'typing' and isinstance(typ, TypeVar)


def isNewType(obj: Any) -> bool:
    """Check the if object is a kind of NewType."""
    if sys.version_info[:2] >= (3, 10):
        return isinstance(obj, typing.NewType)
    __module__ = safe_getattr(obj, '__module__', None)
    __qualname__ = safe_getattr(obj, '__qualname__', None)
    return __module__ == 'typing' and __qualname__ == 'NewType.<locals>.new_type'


def stringify_annotation(annotation: Any, /, mode: str = 'smart') -> str:
    """Stringify type annotation object.

    :param annotation: The annotation to stringified.
    :param mode: Specify a method how annotations will be stringified.

                 'fully-qualified-except-typing'
                     Show the module name and qualified name of the annotation except
                     the "typing" module.
                 'smart'
                     Show the name of the annotation.
                 'fully-qualified'
                     Show the module name and qualified name of the annotation.
    """

    if mode not in {'fully-qualified-except-typing', 'fully-qualified', 'smart'}:
        msg = ("'mode' must be one of 'fully-qualified-except-typing', "
               f"'fully-qualified', or 'smart'; got {mode!r}.")
        raise ValueError(msg)

    if mode == 'smart':
        module_prefix = '~'
    else:
        module_prefix = ''

    annotation_qualname = getattr(annotation, '__qualname__', '')
    annotation_module = getattr(annotation, '__module__', '')
    annotation_name = getattr(annotation, '__name__', '')
    annotation_module_is_typing = annotation_module == 'typing'

    if isinstance(annotation, str):
        if annotation.startswith("'") and annotation.endswith("'"):
            # might be a double Forward-ref'ed type.  Go unquoting.
            return annotation[1:-1]
        else:
            return annotation
    elif isinstance(annotation, TypeVar):
        if annotation_module_is_typing and mode in {'fully-qualified-except-typing', 'smart'}:
            return annotation_name
        else:
            return module_prefix + f'{annotation_module}.{annotation_name}'
    elif isNewType(annotation):
        if sys.version_info[:2] >= (3, 10):
            # newtypes have correct module info since Python 3.10+
            return module_prefix + f'{annotation_module}.{annotation_name}'
        else:
            return annotation_name
    elif not annotation:
        return repr(annotation)
    elif annotation is NoneType:
        return 'None'
    elif is_invalid_builtin_class(annotation):
        return module_prefix + _INVALID_BUILTIN_CLASSES[annotation]
    elif str(annotation).startswith('typing.Annotated'):  # for py310+
        pass
    elif annotation_module == 'builtins' and annotation_qualname:
        if (args := getattr(annotation, '__args__', None)) is not None:  # PEP 585 generic
            if not args:  # Empty tuple, list, ...
                return repr(annotation)

            concatenated_args = ', '.join(stringify_annotation(arg, mode) for arg in args)
            return f'{annotation_qualname}[{concatenated_args}]'
        else:
            return annotation_qualname
    elif annotation is Ellipsis:
        return '...'

    module_prefix = f'{annotation_module}.'
    annotation_forward_arg = getattr(annotation, '__forward_arg__', None)
    if annotation_qualname or (annotation_module_is_typing and not annotation_forward_arg):
        if mode == 'smart':
            module_prefix = '~' + module_prefix
        if annotation_module_is_typing and mode == 'fully-qualified-except-typing':
            module_prefix = ''
    else:
        module_prefix = ''

    if annotation_module_is_typing:
        if annotation_forward_arg:
            # handle ForwardRefs
            qualname = annotation_forward_arg
        else:
            _name = getattr(annotation, '_name', '')
            if _name:
                qualname = _name
            elif annotation_qualname:
                qualname = annotation_qualname
            else:
                qualname = stringify_annotation(
                    annotation.__origin__, 'fully-qualified-except-typing',
                ).replace('typing.', '')  # ex. Union
    elif annotation_qualname:
        qualname = annotation_qualname
    elif hasattr(annotation, '__origin__'):
        # instantiated generic provided by a user
        qualname = stringify_annotation(annotation.__origin__, mode)
    elif UnionType and isinstance(annotation, UnionType):  # types.UnionType (for py3.10+)
        qualname = 'types.UnionType'
    else:
        # we weren't able to extract the base type, appending arguments would
        # only make them appear twice
        return repr(annotation)

    annotation_args = getattr(annotation, '__args__', None)
    if annotation_args:
        if not isinstance(annotation_args, (list, tuple)):
            # broken __args__ found
            pass
        elif qualname in {'Optional', 'Union', 'types.UnionType'}:
            return ' | '.join(stringify_annotation(a, mode) for a in annotation_args)
        elif qualname == 'Callable':
            args = ', '.join(stringify_annotation(a, mode) for a in annotation_args[:-1])
            returns = stringify_annotation(annotation_args[-1], mode)
            return f'{module_prefix}Callable[[{args}], {returns}]'
        elif qualname == 'Literal':
            def isenumattribute(x: Any) -> bool:
                """Check if the object is attribute of enum."""
                return isinstance(x, enum.Enum)

            def format_literal_arg(arg: Any) -> str:
                if isenumattribute(arg):
                    enumcls = arg.__class__

                    if mode == 'smart':
                        # MyEnum.member
                        return f'{enumcls.__qualname__}.{arg.name}'

                    # module.MyEnum.member
                    return f'{enumcls.__module__}.{enumcls.__qualname__}.{arg.name}'
                return repr(arg)

            args = ', '.join(map(format_literal_arg, annotation_args))
            return f'{module_prefix}Literal[{args}]'
        elif str(annotation).startswith('typing.Annotated'):  # for py39+
            return stringify_annotation(annotation_args[0], mode)
        elif all(is_system_TypeVar(a) for a in annotation_args):
            # Suppress arguments if all system defined TypeVars (ex. Dict[KT, VT])
            return module_prefix + qualname
        else:
            args = ', '.join(stringify_annotation(a, mode) for a in annotation_args)
            return f'{module_prefix}{qualname}[{args}]'

    return module_prefix + qualname
