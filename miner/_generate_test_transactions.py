import blockchain
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives import hashes
import random

with open("keys/private_key.bin", "rb") as f, open("keys/public_key.bin", "rb") as g:
    private_key_sender = f.read()
    public_key_sender = g.read()


def new_transaction(amount: int = 10):
    uuid_bytes = random.randbytes(blockchain.Transaction.BYTE_COUNTS[4])
    # generate signature by concatenating sender, receiver, amount, uuid and signing with private key
    _, public_key_receiver = None, random.randbytes(512)
    amount_bytes = amount.to_bytes(blockchain.Transaction.BYTE_COUNTS[2], blockchain.ENDIAN)
    transaction_fee_bytes = (0).to_bytes(blockchain.Transaction.BYTE_COUNTS[3], blockchain.ENDIAN)
    private_key_obj = blockchain.deserialize_private_key(private_key_sender)
    signature_bytes = private_key_obj.sign(
        public_key_sender + public_key_receiver + amount_bytes + transaction_fee_bytes + uuid_bytes,
        padding.PKCS1v15(), hashes.SHA256())
    return blockchain.Transaction(public_key_sender, public_key_receiver, amount, 0, uuid_bytes,
                                  signature_bytes).to_bytes().hex()


if __name__ == "__main__":
    for i in range(10):
        print(new_transaction(), "\n\n\n")
