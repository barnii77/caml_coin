import blockchain
import random

with open("keys/private_key.bin", "rb") as f, open("keys/public_key.bin", "rb") as g:
    private_key_sender = f.read()
    public_key_sender = g.read()


def new_transaction(amount: int = 10):
    uuid_bytes = random.randbytes(blockchain.Transaction.BYTE_COUNTS[4])
    # generate signature by concatenating sender, receiver, amount, uuid and signing with private key
    _, public_key_receiver = blockchain.gen_key_pair()
    amount_bytes = amount.to_bytes(
        blockchain.Transaction.BYTE_COUNTS[2], blockchain.ENDIAN
    )
    transaction_fee_bytes = (0).to_bytes(
        blockchain.Transaction.BYTE_COUNTS[3], blockchain.ENDIAN
    )
    signature_bytes = blockchain.sign_message(
        public_key_sender
        + public_key_receiver
        + amount_bytes
        + transaction_fee_bytes
        + uuid_bytes,
        private_key_sender,
    )
    return (
        blockchain.Transaction(
            public_key_sender,
            public_key_receiver,
            amount,
            0,
            uuid_bytes,
            signature_bytes,
        )
        .to_bytes()
    )


if __name__ == "__main__":
    for i in range(10):
        t = new_transaction()
        h = blockchain.sha256(t)
        print("Transaction:", t.hex())
        print("Hash:", h.hex())
        print("\n")
