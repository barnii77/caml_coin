import functools
import threading

from typing import Generic, TypeVar

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.backends import default_backend

NANOSECOND = 0.000000001
T = TypeVar("T")


def thread_yield():
    """Makes thread scheduler yield control (avoids busy waiting)."""
    pass  # time.sleep(5 * NANOSECOND)


def sha256(x: bytes) -> bytes:
    digest = hashes.Hash(hashes.SHA256(), backend=default_backend())
    digest.update(x)
    return digest.finalize()[::-1]  # little endian output


def broken(reason):
    """Decorator to mark a function as broken."""

    def decorator(func):
        @functools.wraps(func)
        def new_func(*args, **kwargs):
            raise RuntimeError(reason)

        return new_func

    return decorator


# def set_shared_methods(shared: "Shared"):
#     # all methods of wrapped object (e.g. __add__) will be transferred out
#     method_names = list(filter(lambda attr: callable(getattr(shared.x, attr)) and attr not in ("__init__", "__new__", "__del__", "x"), dir(shared.x)))
#     for m in method_names:
#         setattr(shared, m, getattr(shared.x, m))


class Shared(Generic[T]):
    def __init__(self, x: T):
        self._x = x
        # set_shared_methods(self)

    @property
    def x(self):
        return self._x

    @x.setter
    def x(self, value):
        self._x = value
        # set_shared_methods(self)


class LinkedEvent(threading.Event):
    def __init__(self, e1, e2):
        super().__init__()
        self.e1 = e1
        self.e2 = e2

    def is_set(self):
        return self.e1.is_set() or self.e2.is_set()
