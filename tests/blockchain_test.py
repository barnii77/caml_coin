import traceback
import threading

from miner import blockchain
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives import hashes

USE_CUDA = True
if USE_CUDA:
    blockchain.load_cuda_miner("miner/miner.cu")

_test_registry = []


def make_test(correct_answer, notify_on_passed=True, print_result=False):
    def decorator(func):
        def wrapper():
            name = func.__name__
            passed = False
            try:
                out = func()
            except Exception as e:
                print(f"Test {name} failed with exception: " + str(e))
                traceback.print_exc()
            else:
                if out == correct_answer:
                    passed = True
                    if notify_on_passed:
                        print(f"Test {name} passed")
                else:
                    print(f"Test {name} failed")
                if print_result:
                    print(f"OUTPUT of {name}:\n", out, sep="")
            return passed

        return wrapper

    return decorator


def register_test(func):
    _test_registry.append(func)
    return func


class TestThread(threading.Thread):
    def __init__(self, target):
        super().__init__()
        self.target = target
        self.passed = False

    def run(self):
        self.passed = self.target()


def run_tests():
    threads = []
    for test in _test_registry:
        thread = TestThread(target=test)
        thread.start()
        threads.append(thread)
    for thread in threads:
        thread.join()

    all_passed = True
    for thread in threads:
        all_passed = all_passed and thread.passed
    if all_passed:
        print("All tests passed")
    else:
        print("Some tests failed")


@register_test
@make_test(correct_answer=[11, 82, 624, 94, 1000])
def test_blockchain_raw1():
    import random
    import threading

    names_to_keys = {b"name1": blockchain.gen_key_pair(), b"name2": blockchain.gen_key_pair(),
                     b"name4": blockchain.gen_key_pair(), b"name3": blockchain.gen_key_pair(),
                     b"name5": blockchain.gen_key_pair()}

    def name_to_private_public_key(name: bytes):
        return names_to_keys[name]

    init_balance = {name_to_private_public_key(name)[1]: value for name, value in
                    {b"name1": 101, b"name2": 2, b"name4": 624, b"name3": 84, b"name5": 1000}.items()}

    in_tq, in_cq, out_cq, out_bq, out_sq = [], [], [], [], []
    terminate_event = threading.Event()
    blockchain_args = (
        name_to_private_public_key(b"name5")[1], None, in_tq, in_cq, out_cq, out_bq, out_sq, init_balance, terminate_event)

    blockchain_thread = threading.Thread(target=blockchain._mine, args=blockchain_args)
    blockchain_thread.start()

    def get_test_transaction(sender: bytes, receiver: bytes, amount: int):
        uuid_bytes = random.randbytes(blockchain.Transaction.BYTE_COUNTS[4])
        # generate signature by concatenating sender, receiver, amount, uuid and signing with private key
        private_key_sender, public_key_sender = name_to_private_public_key(sender)
        _, public_key_receiver = name_to_private_public_key(receiver)
        amount_bytes = amount.to_bytes(blockchain.Transaction.BYTE_COUNTS[2], blockchain.ENDIAN)
        transaction_fee_bytes = (0).to_bytes(blockchain.Transaction.BYTE_COUNTS[3], blockchain.ENDIAN)
        private_key_obj = blockchain.deserialize_private_key(private_key_sender)
        signature_bytes = private_key_obj.sign(
            public_key_sender + public_key_receiver + amount_bytes + transaction_fee_bytes + uuid_bytes,
            padding.PKCS1v15(), hashes.SHA256())
        return blockchain.Transaction(public_key_sender, public_key_receiver, amount, 0, uuid_bytes, signature_bytes)

    def test_transaction(sender: bytes, receiver: bytes):
        transaction = get_test_transaction(sender, receiver, 10)
        transaction_bytes = transaction.to_bytes()
        in_tq.append(transaction_bytes)

    test_transaction(b"name1", b"name2")
    test_transaction(b"name1", b"name2")
    test_transaction(b"name1", b"name2")
    test_transaction(b"name1", b"name2")
    test_transaction(b"name1", b"name2")
    test_transaction(b"name1", b"name2")
    test_transaction(b"name1", b"name2")
    test_transaction(b"name1", b"name2")
    test_transaction(b"name1", b"name2")
    test_transaction(b"name2", b"name1")
    test_transaction(b"name1", b"name2")
    test_transaction(b"name2", b"name3")

    import time
    time.sleep(5)
    terminate_event.set()
    blockchain_thread.join()
    balances = {hex(int.from_bytes(key, blockchain.ENDIAN))[:20]: value for key, value in out_bq.pop().items()}
    balances_values = list(balances.values())
    return balances_values


@register_test
@make_test([11, 82, 624, 94, 1000])
def test_blockchain_class1():
    import random

    names_to_keys = {b"name1": blockchain.gen_key_pair(), b"name2": blockchain.gen_key_pair(),
                     b"name4": blockchain.gen_key_pair(), b"name3": blockchain.gen_key_pair(),
                     b"name5": blockchain.gen_key_pair()}

    def name_to_private_public_key(name: bytes):
        if name not in names_to_keys:
            names_to_keys[name] = blockchain.gen_key_pair()
        return names_to_keys[name]

    init_balance = {name_to_private_public_key(name)[1]: value for name, value in
                    {b"name1": 101, b"name2": 2, b"name4": 624, b"name3": 84, b"name5": 1000}.items()}

    chain_handler = blockchain.Miner(name_to_private_public_key(b"name5")[1], None, init_balance,
                                     blockchain.MiningConfig(), "threading")
    chain_handler.use_cuda_miner()

    def get_test_transaction(sender: bytes, receiver: bytes, amount: int):
        uuid_bytes = random.randbytes(blockchain.Transaction.BYTE_COUNTS[4])
        # generate signature by concatenating sender, receiver, amount, uuid and signing with private key
        private_key_sender, public_key_sender = name_to_private_public_key(sender)
        _, public_key_receiver = name_to_private_public_key(receiver)
        amount_bytes = amount.to_bytes(blockchain.Transaction.BYTE_COUNTS[2], blockchain.ENDIAN)
        transaction_fee_bytes = (0).to_bytes(blockchain.Transaction.BYTE_COUNTS[3], blockchain.ENDIAN)
        private_key_obj = blockchain.deserialize_private_key(private_key_sender)
        signature_bytes = private_key_obj.sign(
            public_key_sender + public_key_receiver + amount_bytes + transaction_fee_bytes + uuid_bytes,
            padding.PKCS1v15(), hashes.SHA256())
        return blockchain.Transaction(public_key_sender, public_key_receiver, amount, 0, uuid_bytes, signature_bytes)

    def test_transaction(sender: bytes, receiver: bytes):
        transaction = get_test_transaction(sender, receiver, 10)
        transaction_bytes = transaction.to_bytes()
        chain_handler.add_transaction(transaction_bytes)

    test_transaction(b"name1", b"name2")
    test_transaction(b"name1", b"name2")
    test_transaction(b"name1", b"name2")
    test_transaction(b"name1", b"name2")
    test_transaction(b"name1", b"name2")
    test_transaction(b"name1", b"name2")
    test_transaction(b"name1", b"name2")
    test_transaction(b"name1", b"name2")
    test_transaction(b"name1", b"name2")
    test_transaction(b"name2", b"name1")
    test_transaction(b"name1", b"name2")
    test_transaction(b"name2", b"name3")

    import time
    time.sleep(5)
    _, balances = chain_handler.finish()
    balances = blockchain.deserialize_balances(balances)
    balances = {hex(int.from_bytes(key, blockchain.ENDIAN))[:20]: value for key, value in balances.items()}
    balances_values = list(balances.values())
    return balances_values


@register_test
@make_test([1, 82, 634, 94, 1000])
def test_blockchain_class2():
    import random

    names_to_keys = {b"name1": blockchain.gen_key_pair(), b"name2": blockchain.gen_key_pair(),
                     b"name4": blockchain.gen_key_pair(), b"name3": blockchain.gen_key_pair(),
                     b"name5": blockchain.gen_key_pair()}

    def name_to_private_public_key(name: bytes):
        if name not in names_to_keys:
            names_to_keys[name] = blockchain.gen_key_pair()
        return names_to_keys[name]

    init_balance = {name_to_private_public_key(name)[1]: value for name, value in
                    {b"name1": 101, b"name2": 2, b"name4": 624, b"name3": 84, b"name5": 1000}.items()}

    chain_handler = blockchain.Miner(name_to_private_public_key(b"name5")[1], None, init_balance,
                                     blockchain.MiningConfig(), "threading")
    chain_handler.use_cuda_miner()

    def get_test_transaction(sender: bytes, receiver: bytes, amount: int):
        uuid_bytes = random.randbytes(blockchain.Transaction.BYTE_COUNTS[4])
        # generate signature by concatenating sender, receiver, amount, uuid and signing with private key
        private_key_sender, public_key_sender = name_to_private_public_key(sender)
        _, public_key_receiver = name_to_private_public_key(receiver)
        amount_bytes = amount.to_bytes(blockchain.Transaction.BYTE_COUNTS[2], blockchain.ENDIAN)
        transaction_fee_bytes = (0).to_bytes(blockchain.Transaction.BYTE_COUNTS[3], blockchain.ENDIAN)
        private_key_obj = blockchain.deserialize_private_key(private_key_sender)
        signature_bytes = private_key_obj.sign(
            public_key_sender + public_key_receiver + amount_bytes + transaction_fee_bytes + uuid_bytes,
            padding.PKCS1v15(), hashes.SHA256())
        return blockchain.Transaction(public_key_sender, public_key_receiver, amount, 0, uuid_bytes, signature_bytes)

    def test_transaction(sender: bytes, receiver: bytes, amount=10):
        transaction = get_test_transaction(sender, receiver, amount)
        transaction_bytes = transaction.to_bytes()
        chain_handler.add_transaction(transaction_bytes)

    test_transaction(b"name1", b"name2")
    test_transaction(b"name1", b"name2")
    test_transaction(b"name1", b"name2")
    test_transaction(b"name1", b"name2")
    test_transaction(b"name1", b"name2")
    test_transaction(b"name1", b"name2")
    test_transaction(b"name1", b"name2")
    test_transaction(b"name1", b"name2")
    test_transaction(b"name1", b"name2")
    test_transaction(b"name1", b"name2")
    test_transaction(b"name2", b"name3")
    test_transaction(b"name2", b"name4")
    test_transaction(b"name1", b"name2")  # should fail (not enough coins)
    test_transaction(b"name1", b"name2")  # should fail (not enough coins)
    test_transaction(b"name1", b"name2")  # should fail (not enough coins)
    test_transaction(b"not real name", b"name3")  # should fail (not real name has balance 0)
    test_transaction(b"name2", b"name2")  # should fail (sender and receiver are the same)
    test_transaction(b"name2", b"name3", 100000)  # should fail (not enough coins)
    test_transaction(b"name2", b"name3", 0)  # should fail (amount is 0)

    # test with invalid signature
    transaction = get_test_transaction(b"name1", b"name2", 10)
    transaction.uuid = random.randbytes(blockchain.Transaction.BYTE_COUNTS[4])  # change uuid, now signature is invalid
    transaction_bytes = transaction.to_bytes()
    chain_handler.add_transaction(transaction_bytes)

    # test with duplicated uuid (send same transaction twice)
    transaction = get_test_transaction(b"name1", b"name2", 10)
    transaction_bytes = transaction.to_bytes()
    chain_handler.add_transaction(transaction_bytes)
    chain_handler.add_transaction(transaction_bytes)

    import time
    time.sleep(5)
    _, balances = chain_handler.finish()
    balances = blockchain.deserialize_balances(balances)
    balances = {hex(int.from_bytes(key, blockchain.ENDIAN))[:20]: value for key, value in balances.items()}
    balances_values = list(balances.values())
    return balances_values


if __name__ == "__main__":
    run_tests()