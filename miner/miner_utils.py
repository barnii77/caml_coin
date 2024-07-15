import time

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.backends import default_backend

NANOSECOND = 0.000000001


def thread_yield():
    """Makes thread scheduler yield control (avoids busy waiting)."""
    time.sleep(10 * NANOSECOND)


def sha256(x: bytes) -> bytes:
    digest = hashes.Hash(hashes.SHA256(), backend=default_backend())
    digest.update(x)
    return digest.finalize()[::-1]  # little endian output
