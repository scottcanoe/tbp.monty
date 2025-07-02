import functools
import inspect
import time
from types import MethodType
from typing import Callable, Iterable, List, Optional, Tuple, Union

import numpy as np
import wrapt
from wrapt import ObjectProxy


class A:
    def __init__(self, name: str):
        self.name = name

    def f(self, *args, **kwargs):
        print(f"A.f: {self.name}")
        # return 0

    def g(self, *args, **kwargs):
        print(f"A.g: {self.name}")

    def h(self, x, y):
        return 2
        # return f"{self.__class__.__name__}('{self.name}').h(x={x}, y={y})"


class B(A):
    pass


@wrapt.decorator
def add_one(wrapped, instance, args, kwargs):
    y = args[0]
    # print(f"ADD ONE: instance: {instance}, args: {args}, kwargs: {kwargs}")
    return wrapped(*args, **kwargs) + 1


# @wrapt.decorator
# def add_meta(wrapped, instance, args, kwargs):
#     if wrapped.__dict__.get("meta", False):
#         pass
#         wrapped.__dict__["meta"] = {"add_meta": True}
#         print(f"--add_meta--")
#     return wrapped(*args, **kwargs)

from copy import deepcopy


def inc(wrapper):
    """_summary_

    Returns:
        _type_: _description_
    """

    # if level is not set, set it to 0
    if wrapper.__dict__.get("level") is None:
        wrapper.__dict__["level"] = 0
    else:
        wrapper.__dict__["level"] += 1

    # other case
    # if wrapper.__dict__.get("level") is not None:
    #     print(
    #         f"ALREADY WRAPPED: wrapper: {type(wrapper)}, __wrapped__: {type(wrapper.__wrapped__)}"
    #     )
    #     return wrapper
    # print(f"WRAPPING {wrapper}")
    # wrapper.__dict__.get("level") is None:

    # level = wrapper.__dict__["level"] = 0
    # level = wrapper.__dict__["level"]
    level = wrapper.__dict__["level"]

    @wrapt.decorator
    def fn(wrapped, instance, args, kwargs):
        # print(f"level {level}")
        args = [args[0] + 1]
        return wrapped(*args, **kwargs)

    return fn(wrapper)


def log(wrapped, instance, args, kwargs):
    print(
        f"log: wrapped: {wrapped}, instance: {instance}, args: {args}, kwargs: {kwargs}"
    )


def has_wrapper_tag(obj, tag: str):
    if hasattr(obj, "_wrapper_tags"):
        return True if tag in obj._wrapper_tags else False
    try:
        return tag in obj.__dict__["_wrapper_tags"]
    except KeyError:
        return False


def add_wrapper_tag(obj, tag: str):
    if hasattr(obj, "_wrapper_tags"):
        obj._wrapper_tags.add(tag)
        return
    try:
        obj._wrapper_tags = {tag}
        return
    except AttributeError:
        pass
    if "_wrapper_tags" in obj.__dict__:
        obj.__dict__["_wrapper_tags"].add(tag)
        return
    obj.__dict__["_wrapper_tags"] = {tag}


def wrap_once(tag: str):
    """Prevents a function from being decorated more than once by the same decorator.

    Args:
        tag: ID of decorator.

    Returns:
        A wrapt-compatible decorator.
    """

    def decorator(func):
        def wrapper(obj, *args, **kwargs):
            print(f"1. func: {func} , obj: {obj}")
            if has_wrapper_tag(obj, tag):
                print(f"2. obj: {obj} is already wrapped")
                return obj
            print(f"3. obj: {obj} is not wrapped. apply wrapper")
            result = func(obj, *args, **kwargs)
            print(f"4. result: {result}")
            add_wrapper_tag(result, tag)
            return result

        return wrapper

    return decorator


def wrap_method(obj, method_name: str):
    method = getattr(obj, method_name)
    if has_wrapper_tag(method, "wrap_method"):
        return method
    add_wrapper_tag(method, "wrap_method")

    @wrapt.decorator
    def wrapper(wrapped, instance, args, kwargs):
        print(" + ", end="")
        return wrapped(*args, **kwargs)

    out = wrapper(method, obj)
    return out


@wrap_once("g_thang")
def g_thang(func):
    @wrapt.decorator
    def wrapper(wrapped, instance, args, kwargs):
        print(" G-THANG ", end="")
        return wrapped(*args, **kwargs)

    return wrapper(func)


radius = 60
ndivs = 10

circ = 2 * np.pi * radius
ln = circ / ndivs
