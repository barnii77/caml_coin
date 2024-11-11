import json
import traceback
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.asymmetric import ec

ENDIAN = "little"
PRIVATE_KEY_SIZE = 32
PUBLIC_KEY_SIZE = 33
try:
    with open("data/cc_utils_balances.json") as f:
        balances = json.load(f)
    with open("data/cc_utils_fake_points.json") as f:
        fake_points = json.load(f)
except Exception:
    traceback.format_exc()
    balances = {}
    fake_points = {}


class TooPoorException(Exception):
    pass


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


def send_coins(sender_user_info, receiver_public_key, n):
    if balances.setdefault(sender_user_info.public_key, 0) < n:
        raise TooPoorException("sender does not have enough money")
    balances[receiver_public_key] = balances.get(receiver_public_key, 0) + n
    balances[sender_user_info.public_key] -= n


def get_available_coins(public_key):
    return balances.get(public_key, 0)


def get_available_fake_points(public_key):
    return fake_points.get(public_key, 0)


def withdraw_fake_points(public_key, amount: int) -> int:
    fake_points[public_key] = fake_points.get(public_key, 0) - amount
    return fake_points[public_key]


def deposit_fake_points(public_key, amount: int) -> int:
    fake_points[public_key] = fake_points.get(public_key, 0) + amount
    return fake_points[public_key]
