import threading
import multiprocessing as mp
import queue

from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.backends import default_backend
from typing import Optional

# fix python std lib bug in mp (have to create queue before I can access mp.queues)
mp.Queue()
try:
    mp.Queue()
except AttributeError:
    raise RuntimeError("mp.queues submodule could not be loaded. this is a bug in python std lib... sry :(")

KEY_SIZE = 4096
ENDIAN = "little"
LXX = lambda x: x
VALID_BLOCK_LEADING_ZEROS = 8
CUDA_MINER_PATH = "miner.cu"


def sha256(x: bytes):
    digest = hashes.Hash(hashes.SHA256(), backend=default_backend())
    digest.update(x)
    return digest.finalize()


def is_valid_block(block: "Block"):
    if block is None:
        return True
    h = block.hash()
    for byte in h[:VALID_BLOCK_LEADING_ZEROS // 8]:
        if byte != 0:
            return False
    return h[(VALID_BLOCK_LEADING_ZEROS + 7) // 8].bit_length() <= 8 - VALID_BLOCK_LEADING_ZEROS % 8


def is_valid_chain(block: "Block"):
    if block is None:
        return True
    while block.prev is not None:
        if not is_valid_block(block):
            return False
        block = block.prev
    return True


def deserialize_private_key(key: bytes):
    p_bytes, q_bytes, d_bytes = key[:KEY_SIZE // 16], key[KEY_SIZE // 16:KEY_SIZE // 8], key[KEY_SIZE // 8:]
    p, q, d = int.from_bytes(p_bytes, ENDIAN), int.from_bytes(q_bytes, ENDIAN), int.from_bytes(d_bytes, ENDIAN)
    private_key = rsa.RSAPrivateNumbers(p=p, q=q, d=d, dmp1=d % (p - 1), dmq1=d % (q - 1), iqmp=rsa.rsa_crt_iqmp(p, q),
                                        public_numbers=rsa.RSAPublicNumbers(e=65537, n=p * q)).private_key(
        default_backend())
    return private_key


def deserialize_public_key(key: bytes):
    public_numbers = int.from_bytes(key, ENDIAN)
    public_key = rsa.RSAPublicNumbers(e=65537, n=public_numbers).public_key(default_backend())
    return public_key


def serialize_private_key(key):
    private_numbers = key.private_numbers()
    private_bytes = (private_numbers.p.to_bytes(KEY_SIZE // 16, ENDIAN) + private_numbers.q.to_bytes(KEY_SIZE // 16,
                                                                                                     ENDIAN) + private_numbers.d.to_bytes(
        KEY_SIZE // 8, ENDIAN))
    return private_bytes


def serialize_public_key(key):
    public_numbers = key.public_numbers()
    public_bytes = public_numbers.n.to_bytes(KEY_SIZE // 8, ENDIAN)
    return public_bytes


def sign_message(message: bytes, private_key_bytes: bytes):
    private_key = deserialize_private_key(private_key_bytes)
    signature = private_key.sign(message, padding.PKCS1v15(), hashes.SHA256())
    return signature


def verify_signature(message: bytes, signature: bytes, public_key_bytes: bytes):
    public_key = deserialize_public_key(public_key_bytes)
    try:
        # Verify the signature
        public_key.verify(signature, message, padding.PKCS1v15(), hashes.SHA256())
    except Exception:
        return False
    return True


def chain_len(block: "Block"):
    if block is None:
        return 0
    x = 1
    while block.prev is not None:
        block = block.prev
        x += 1
    return x


def execute_transaction(t: "Transaction", validator: bytes, balances: dict[bytes, int], invalidated_uuids: set[bytes]):
    balances[t.sender] = balances.get(t.sender, 0) - t.amount
    balances[t.receiver] = balances.get(t.receiver, 0) + t.amount - t.transaction_fee
    balances[validator] = balances.get(validator, 0) + t.transaction_fee
    invalidated_uuids.add(t.uuid)


def full_verify(block: "Block", init_balance: dict[bytes, int], val_reward: int, const_transaction_fee: int):
    balances = init_balance.copy()
    invalidated_uuids = set()
    if block is None:
        return True, balances, invalidated_uuids
    if not is_valid_chain(block):
        return False, balances, invalidated_uuids

    while block is not None:
        for t in block.transactions:
            if not t.is_valid(balances, invalidated_uuids, const_transaction_fee):
                return False, balances, invalidated_uuids
            execute_transaction(t, block.validator, balances, invalidated_uuids)
        balances[block.validator] = balances.get(block.validator, 0) + val_reward
        block = block.prev
    return True, balances, invalidated_uuids


class Transaction:
    BYTE_COUNTS = [KEY_SIZE // 8, KEY_SIZE // 8, 8, 8, 16, KEY_SIZE // 8]
    N_BYTES = sum(BYTE_COUNTS)
    CONVERSION_F = [LXX, LXX, lambda b: int.from_bytes(b, ENDIAN), lambda b: int.from_bytes(b, ENDIAN), LXX, LXX]

    def __init__(self, sender: bytes, receiver: bytes, amount: int, transaction_fee: int, uuid: bytes,
                 signature: bytes):
        self.sender = sender
        self.receiver = receiver
        self.amount = amount
        self.transaction_fee = transaction_fee
        self.uuid = uuid
        self.signature = signature

    def _bytes_without_signature(self):
        amount = self.amount.to_bytes(Transaction.BYTE_COUNTS[2], ENDIAN)
        transaction_fee = self.transaction_fee.to_bytes(Transaction.BYTE_COUNTS[3], ENDIAN)
        return self.sender + self.receiver + amount + transaction_fee + self.uuid

    def to_bytes(self):
        return self._bytes_without_signature() + self.signature

    def is_valid(self, balances: dict[to_bytes, int], invalidated_uuids: set[to_bytes], const_transaction_fee: int):
        can_afford = balances.get(self.sender, 0) > self.amount > 0
        signature_valid = verify_signature(self._bytes_without_signature(), self.signature, self.sender)
        uuid_valid = self.uuid not in invalidated_uuids
        has_correct_fee = self.transaction_fee == const_transaction_fee
        return signature_valid and can_afford and uuid_valid and self.amount >= self.transaction_fee and has_correct_fee

    @classmethod
    def from_bytes(cls, raw: to_bytes):
        components = []
        cum_size = 0
        for size, cf in zip(Transaction.BYTE_COUNTS, Transaction.CONVERSION_F):
            components.append(cf(raw[cum_size:cum_size + size]))
            cum_size += size
        return cls(*components)


class Block:
    SIZE = 4
    BYTE_COUNTS = [32, 32, 32] + [Transaction.N_BYTES] * SIZE
    N_BYTES = sum(BYTE_COUNTS)
    CONVERSION_F = [lambda b: int.from_bytes(b, ENDIAN), LXX, LXX] + [Transaction.from_bytes] * SIZE

    def __init__(self, transactions: list["Transaction"], prev: "Block", validator: bytes, validation: int = 0):
        self.validator = validator
        self.transactions = transactions
        self.prev = prev
        self.validation = validation
        self._cache_hash = None
        self._cache_inner_hash = None
        self._cache_val = validation

    @classmethod
    def find_val(cls, open_block: "OpenBlock", validator: bytes, use_cuda_miner: bool = False):
        if use_cuda_miner:
            raise NotImplementedError("CUDA miner not implemented")
        else:
            block = cls(open_block.transactions, open_block.prev, validator, 0)
            while not is_valid_block(block):
                block.validation += 1
            return block

    def _bytes_without_val(self):
        return (
            self.prev.hash() if self.prev is not None else b'\0' * Block.BYTE_COUNTS[1]) + self.validator + b''.join(
            map(Transaction.to_bytes, self.transactions))

    def to_bytes(self):
        return self.validation.to_bytes(Block.BYTE_COUNTS[0], ENDIAN) + self._bytes_without_val()

    @classmethod
    def from_bytes(cls, raw: to_bytes):
        # NOTE: method assigns prev hash to prev member that should be Block object (gets linked externally)
        components = []
        cum_size = 0
        for size, cf in zip(Transaction.BYTE_COUNTS, Transaction.CONVERSION_F):
            components.append(cf(raw[cum_size:cum_size + size]))
            cum_size += size

        # last Block.SIZE components are transactions
        t_start = sum(Block.BYTE_COUNTS[:-Block.SIZE])
        components, transactions = components[:t_start], components[t_start:]
        components.insert(0, transactions)
        return cls(*components)

    def hash(self):
        if self._cache_hash is None or self.validation != self._cache_val:
            if self._cache_inner_hash is None:
                self._cache_inner_hash = sha256(self._bytes_without_val())
            self._cache_hash = sha256(self.validation.to_bytes(Block.BYTE_COUNTS[0], ENDIAN) + self._cache_inner_hash)
            self._cache_val = self.validation
        return self._cache_hash


def chain_from_bytes(raw: bytes):
    if raw == b'':
        return None
    block_bytes = [raw[i:i + Block.N_BYTES] for i in
                   range(0, int(len(raw) / Block.N_BYTES) * Block.N_BYTES, Block.N_BYTES)]
    blocks = list(map(Block.from_bytes, block_bytes))
    # link blocks
    for block in blocks:
        if block.prev == b'\0' * len(block.prev):
            break
    else:
        # no origin block
        return None
    block.prev = None
    blocks.remove(block)
    while blocks:
        h = block.hash()
        prevs = list(map(lambda b: b.prev, blocks))
        if h not in prevs:
            return None  # no matching next block
        next_block = blocks[prevs.index(h)]
        next_block.prev = block
        blocks.remove(block)
        block = next_block
    return block


class OpenBlock:
    def __init__(self, prev: Optional["Block"], validator: bytes, transactions: list["Transaction"] = None):
        if transactions is None:
            transactions = []
        self.validator = validator
        self.transactions = transactions
        self.prev = prev

    def receive(self, transaction: "Transaction", balances: dict[bytes, int], invalidated_uuids: set[bytes],
                const_transaction_fee: int, val_reward: int):
        if not transaction.is_valid(balances, invalidated_uuids, const_transaction_fee):
            return False, False
        self.transactions.append(transaction)
        execute_transaction(transaction, self.validator, balances, invalidated_uuids)
        if len(self.transactions) < Block.SIZE:
            return False, True
        else:
            balances[self.validator] = balances.get(self.validator, 0) + val_reward
            return True, True


def get_received_transactions(transaction_buffer, n: int):
    if transaction_buffer is None:
        return []
    out = []
    if isinstance(transaction_buffer, (queue.Queue, mp.queues.Queue)):
        while not transaction_buffer.empty() and n:
            try:
                raw = transaction_buffer.get_nowait()
            except queue.Empty:
                pass
            else:
                if len(raw) != Transaction.N_BYTES:
                    continue
                out.append(Transaction.from_bytes(raw))
                n -= 1
    else:
        while n and transaction_buffer:
            raw = transaction_buffer.pop(0)
            if len(raw) != Transaction.N_BYTES:
                continue
            out.append(Transaction.from_bytes(raw))
            n -= 1
    return out


def get_received_chains(chain_buffer, n: int):
    if chain_buffer is None:
        return []
    out = []
    if isinstance(chain_buffer, (queue.Queue, mp.queues.Queue)):
        while not chain_buffer.empty() and n:
            try:
                raw = chain_buffer.get_nowait()
            except queue.Empty:
                pass
            else:
                if len(raw) != Block.N_BYTES:
                    continue
                chain = chain_from_bytes(raw)
                if chain is not None:
                    out.append(chain)
                    n -= 1
    else:
        while n and chain_buffer:
            raw = chain_buffer.pop(0)
            if len(raw) != Block.N_BYTES:
                continue
            chain = chain_from_bytes(raw)
            if chain is not None:
                out.append(chain)
                n -= 1
    return out


def broadcast_chain(broadcast_buffer, block: "Block"):
    if broadcast_buffer is None:
        return
    if block is not None:
        bb = block.to_bytes()
        if isinstance(broadcast_buffer, (queue.Queue, mp.queues.Queue)):
            try:
                broadcast_buffer.put_nowait(bb)
            except queue.Full:
                broadcast_buffer.get_nowait()
                broadcast_buffer.put_nowait(bb)
        else:
            broadcast_buffer.append(bb)


def broadcast_balances(broadcast_buffer, balances: dict[bytes, int], copy: bool = False):
    if broadcast_buffer is None:
        return
    if isinstance(broadcast_buffer, (queue.Queue, mp.queues.Queue)):
        try:
            broadcast_buffer.put_nowait(balances.copy() if copy else balances)
        except queue.Full:
            broadcast_buffer.get_nowait()
            broadcast_buffer.put_nowait(balances.copy() if copy else balances)
    else:
        broadcast_buffer.append(balances.copy() if copy else balances)


def blockchain(validator: bytes, chain: "Block" = None, transaction_recv_buffer=None, chain_recv_buffer=None,
               chain_broadcast_buffer=None, balance_broadcast_buffer=None, init_balance=None, terminate_event=None,
               has_terminated_event=None, use_cuda_miner_event=None, incompatible_chain_distrust=5,
               compatible_chain_distrust=1, val_reward: int = 1, const_transaction_fee: int = 0,
               max_recv_chains_per_iter: int = -1, max_recv_transactions_per_iter: int = -1,
               copy_balances_on_broadcast=False):
    """
    The main blockchain function. It receives transactions and chains, verifies them, and updates the blockchain.
    :param validator: the public key of the miner
    :param chain: optional initial chain
    :param transaction_recv_buffer: queue for receiving transactions
    :param chain_recv_buffer: queue for receiving chains
    :param chain_broadcast_buffer: queue for broadcasting new chains that a validation was found for
    :param balance_broadcast_buffer: queue for broadcasting balances for review
    :param init_balance: initial balance of the blockchain (what the balance is before the first transaction)
    :param terminate_event: event set when the blockchain main loop should terminate
    :param has_terminated_event: event set when the blockchain main loop has terminated
    :param use_cuda_miner_event: event that signals that the miner should use the CUDA miner kernel instead of the debug miner
    :param incompatible_chain_distrust: how many blocks a chain that has incompatible balances with the current chain must be longer to replace the current chain
    :param compatible_chain_distrust: how many blocks a chain that has the same balances at the end has to be longer to replace the current chain
    :param val_reward: the reward for validating a block
    :param const_transaction_fee: the constant transaction fee that is subtracted from the amount sent and given to the validator
    :param max_recv_chains_per_iter: how many alternative chains may be processed in one loop iteration (helps limit time / iter)
    :param max_recv_transactions_per_iter: how many transactions may be processed in one loop iteration (helps limit time / iter)
    :param copy_balances_on_broadcast: whether to copy the balances dictionary when broadcasting it (helps prevent bugs by removing mutation coupling with blockchain core)
    """
    if init_balance is None:
        init_balance = {}
    open_block, balances, invalidated_uuids = OpenBlock(chain, validator), init_balance, set()
    while terminate_event is None or not terminate_event.is_set():
        cln = chain_len(chain)
        updated_chain = False

        # receive new chains
        recv_chains = get_received_chains(chain_recv_buffer, max_recv_chains_per_iter)
        favorite_new_chain_data = (None, None, None)
        max_trust_level = 0
        for recv_chain in recv_chains:
            ncl = chain_len(recv_chain)
            is_valid, new_balances, new_invalidated_uuids = full_verify(recv_chain, init_balance, val_reward,
                                                                        const_transaction_fee)
            # if chains are equivalent, trust them more than if the result is different
            if is_valid and (ncl - cln - incompatible_chain_distrust > max_trust_level or (
                    ncl - cln - compatible_chain_distrust > max_trust_level and new_balances == balances)):
                favorite_new_chain_data = (recv_chain, new_balances, new_invalidated_uuids)
                max_trust_level = max(ncl - cln - incompatible_chain_distrust, ncl - cln - compatible_chain_distrust)
        if favorite_new_chain_data[0] is not None:
            chain, balances, invalidated_uuids = favorite_new_chain_data
            updated_chain = True

        # receive new transactions
        recv_trans = sorted(get_received_transactions(transaction_recv_buffer, max_recv_transactions_per_iter),
                            key=lambda t: t.amount, reverse=True)
        for t in recv_trans:
            is_closed, is_updated = open_block.receive(t, balances, invalidated_uuids, const_transaction_fee,
                                                       val_reward)
            if is_closed:
                chain = Block.find_val(open_block, validator,
                                       use_cuda_miner_event is not None and use_cuda_miner_event.is_set())
                open_block = OpenBlock(chain, validator)

            if is_updated or is_closed:
                updated_chain = True

        if updated_chain:
            broadcast_chain(chain_broadcast_buffer, chain)
            broadcast_balances(balance_broadcast_buffer, balances, copy_balances_on_broadcast)

    if has_terminated_event is not None:
        has_terminated_event.set()


class BlockchainConfig:
    def __init__(self, incompatible_chain_distrust: int = 5, compatible_chain_distrust: int = 0, val_reward: int = 1,
                 const_transaction_fee: int = 0, max_recv_chains_per_iter: int = -1,
                 max_recv_transactions_per_iter: int = -1, copy_balances_on_broadcast: bool = True):
        self.incompatible_chain_distrust = incompatible_chain_distrust
        self.compatible_chain_distrust = compatible_chain_distrust
        self.val_reward = val_reward
        self.const_transaction_fee = const_transaction_fee
        self.max_recv_chains_per_iter = max_recv_chains_per_iter
        self.max_recv_transactions_per_iter = max_recv_transactions_per_iter
        self.copy_balances_on_broadcast = copy_balances_on_broadcast


class Blockchain:
    """
    A class that represents a blockchain. It is a wrapper around the blockchain function that allows for easy
    serialization and deserialization of the blockchain. You can add transactions and alternative chains to the blockchain.
    It allows for easy retrieval of the balances of the blockchain.
    """

    def __init__(self, validator: bytes, chain: "Block" = None, init_balance=None, config: BlockchainConfig = None,
                 run_with: str = "direct"):
        super().__init__()
        self.validator = validator
        self.chain = chain
        self.init_balance = init_balance

        if run_with == "threading":
            self.transaction_recv_buffer = queue.Queue()
            self.chain_recv_buffer = queue.Queue()
            self.chain_broadcast_buffer = queue.Queue()
            self.balance_broadcast_buffer = queue.Queue()
            self.terminate_event = threading.Event()
            self.has_terminated_event = threading.Event()
            self.use_cuda_miner_event = threading.Event()
        elif run_with == "mp":
            self.transaction_recv_buffer = mp.Queue()
            self.chain_recv_buffer = mp.Queue()
            self.chain_broadcast_buffer = mp.Queue()
            self.balance_broadcast_buffer = mp.Queue()
            self.terminate_event = mp.Event()
            self.has_terminated_event = mp.Event()
            self.use_cuda_miner_event = mp.Event()
        else:
            self.transaction_recv_buffer = []
            self.chain_recv_buffer = []
            self.chain_broadcast_buffer = []
            self.balance_broadcast_buffer = []
            self.terminate_event = threading.Event()
            self.use_cuda_miner_event = threading.Event()

        self._chain_queue_offload_queue = []
        self._transaction_queue_offload_queue = []

        self.incompatible_chain_distrust = config.incompatible_chain_distrust
        self.compatible_chain_distrust = config.compatible_chain_distrust
        self.val_reward = config.val_reward
        self.const_transaction_fee = config.const_transaction_fee
        self.max_recv_chains_per_iter = config.max_recv_chains_per_iter
        self.max_recv_transactions_per_iter = config.max_recv_transactions_per_iter
        self.copy_balances_on_broadcast = config.copy_balances_on_broadcast

        self.run_with = run_with
        self._get_balances_last = self.init_balance
        self._get_chain_last = self.chain

        args = (self.validator, self.chain, self.transaction_recv_buffer, self.chain_recv_buffer,
                self.chain_broadcast_buffer, self.balance_broadcast_buffer, self.init_balance, self.terminate_event,
                self.has_terminated_event, self.use_cuda_miner_event, self.incompatible_chain_distrust,
                self.compatible_chain_distrust, self.val_reward, self.const_transaction_fee,
                self.max_recv_chains_per_iter, self.max_recv_transactions_per_iter, self.copy_balances_on_broadcast)
        if run_with == "direct":
            blockchain(*args)
        elif run_with == "threading":
            self._thread = threading.Thread(target=blockchain, args=args)
            self._thread.start()
        elif run_with == "mp":
            self._process = mp.Process(target=blockchain, args=args)
            self._process.start()

    def __del__(self):
        self.kill(True)

    def to_bytes(self):
        return self.validator + self.chain.to_bytes()

    @classmethod
    def from_bytes(cls, raw: bytes):
        validator, chain_bytes = raw[:KEY_SIZE // 8], raw[KEY_SIZE // 8:]
        chain = chain_from_bytes(chain_bytes)
        return cls(validator, chain)

    def _unsafe_clear_all_queues(self):
        while not self.transaction_recv_buffer.empty():
            self.transaction_recv_buffer.get_nowait()
        while not self.chain_recv_buffer.empty():
            self.chain_recv_buffer.get_nowait()
        while not self.chain_broadcast_buffer.empty():
            self.chain_broadcast_buffer.get_nowait()
        while not self.balance_broadcast_buffer.empty():
            self.balance_broadcast_buffer.get_nowait()

    def _make_all_queue_interaction_functions_error(self):
        def raise_error(*args, **kwargs):
            raise RuntimeError("Cannot interact with queues after termination")

        self.get_chain = raise_error
        self.get_balances = raise_error
        self.add_transaction = raise_error
        self.add_chain = raise_error

    def kill(self, block_until_terminated=True):
        self.terminate_event.set()
        self.has_terminated_event.wait()
        self._make_all_queue_interaction_functions_error()
        self._unsafe_clear_all_queues()  # otherwise, mp doesn't terminate
        if self.run_with == "mp" and block_until_terminated and self._process.is_alive():
            self._process.join()
        elif self.run_with == "threading" and block_until_terminated and self._thread.is_alive():
            self._thread.join()

    def end(self) -> tuple[Optional[bytes], Optional[dict[bytes, int]]]:
        final_chain, final_balances = self.get_chain(), self.get_balances()
        self.kill(True)
        return final_chain, final_balances

    @staticmethod
    def _safe_add_to_queue(item, q, offload_queue, has_queue_interface: bool):
        if has_queue_interface:
            try:
                if offload_queue:
                    q.put_nowait(offload_queue.pop(0))
                else:
                    q.put_nowait(item)
            except queue.Full:
                offload_queue.append(item)
        else:
            q.append(item)

    def add_transaction(self, transaction: bytes):
        self._safe_add_to_queue(transaction, self.transaction_recv_buffer, self._transaction_queue_offload_queue,
                                self.run_with != "direct")

    def add_chain(self, chain: bytes):
        self._safe_add_to_queue(chain, self.chain_recv_buffer, self._chain_queue_offload_queue,
                                self.run_with != "direct")

    def get_balances(self):
        if self.run_with in ("mp", "threading"):
            out = None
            while not self.balance_broadcast_buffer.empty():
                out = self.balance_broadcast_buffer.get_nowait()
        else:
            out = None
            if self.balance_broadcast_buffer:
                out = self.balance_broadcast_buffer.pop()
                self.balance_broadcast_buffer.clear()

        if out is not None:
            self._get_balances_last = out
        return self._get_balances_last

    def get_chain(self):
        if self.run_with in ("mp", "threading"):
            out = None
            while not self.chain_broadcast_buffer.empty():
                out = self.chain_broadcast_buffer.get_nowait()
        else:
            out = None
            if self.chain_broadcast_buffer:
                out = self.chain_broadcast_buffer.pop()
                self.chain_broadcast_buffer.clear()

        if out is not None:
            self._get_chain_last = out
        return out

    def use_cuda_miner(self):
        self.use_cuda_miner_event.set()

    def use_naive_miner(self):
        self.use_cuda_miner_event.clear()


def gen_key_pair() -> tuple[bytes, bytes]:
    """
    Generate a key pair using RSA with 65537 as the public exponent and KEY_SIZE as the key size.
    :return: (private_key_bytes, public_key_bytes)
    """
    private_key = rsa.generate_private_key(65537, KEY_SIZE)
    private_bytes = serialize_private_key(private_key)
    public_key = private_key.public_key()
    public_bytes = serialize_public_key(public_key)

    return private_bytes, public_bytes
