from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives import serialization

ENDIAN = "little"
PRIVATE_KEY_SIZE = 32
PUBLIC_KEY_SIZE = 33


def generate_ecdsa_key_pair() -> tuple[bytes, bytes]:
    private_key = ec.generate_private_key(ec.SECP256K1(), default_backend())
    private_key_bytes = private_key.private_numbers().private_value.to_bytes(32, byteorder='little')
    public_key = private_key.public_key()
    public_numbers = public_key.public_numbers()
    x = public_numbers.x
    y = public_numbers.y
    if y % 2 == 0:
        y_sign = b'\x02'
    else:
        y_sign = b'\x03'
    public_key_bytes = y_sign + x.to_bytes(32, byteorder='little')
    return private_key_bytes, public_key_bytes


def send_coins(*_):
    pass  # TODO


# TODO make this under the hood have a cache and a thread that overwrites the cache in the background
def get_available_coins(*_):
    pass  # TODO
