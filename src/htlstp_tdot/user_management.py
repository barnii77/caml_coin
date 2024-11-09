from functools import lru_cache
import cc_utils

user_keys_by_id = {}


def get_user_keys(user_id):
    if user_id not in user_keys_by_id:
        user_keys_by_id[user_id] = cc_utils.generate_ecdsa_key_pair()
    priv, pub = user_keys_by_id[user_id]
    return {"private_key": priv, "public_key": pub}


class CamlCoinUserInfo:
    def __init__(self, user_id, private_key: bytes, public_key: bytes):
        if (
            len(public_key) != cc_utils.PUBLIC_KEY_SIZE
            or len(private_key) != cc_utils.PRIVATE_KEY_SIZE
        ):
            raise ValueError(
                f"public key and private key must be 33 bytes and 32 bytes in length respectively, but are {len(public_key)} and {len(private_key)} bytes long"
            )
        self.user_id = user_id
        self.private_key = private_key
        self.public_key = public_key


@lru_cache(maxsize=None)
def get_user_info(user_id):
    keys = get_user_keys(user_id)
    return CamlCoinUserInfo(user_id, keys["private_key"], keys["public_key"])
