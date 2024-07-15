import hashlib
import threading
import multiprocessing as mp
import queue
import enum
import time

from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.backends import default_backend
from typing import Optional, Union, Any, List, Dict

from miner.miner_utils import thread_yield, sha256

_cuda_miner_mod = None  # will be set if needed
_cuda_miner_fixed_launch_config_kernel = None

# fix python std lib bug in mp (have to create queue before I can access mp.queues and same with event and mp.synchronize)
try:
    mp.Queue()
    mp.Event()
except AttributeError:
    raise RuntimeError(
        "mp.queues or mp.synchronize submodule could not be loaded. This is a weirdness in python stdlib... sry :("
    )

KEY_SIZE = 4096
NONCE_SIZE_BYTES = 16  # TODO make 16 when emulated kernel works
ENDIAN = "little"
LXX = lambda x: x
BALANCE_SERIALIZATION_VALUE_SIZE = 16
RUN_WEAKNESS_DETECTION_WARNINGS = True
HAS_VAL_REWARD_WARNING = """Warning, val_reward > 0.
Make sure miners can't do a self-reward attack.
A self-reward attack is when a miner spams transactions from one of his accounts to another (without broadcasting) and mines on those transactions.
This way he can get the reward without actually doing anything (useful)."""


def load_cuda_miner(path="miner_template.cu", autotune=True, show_progress=False):
    import miner.cuda_miner as cuda_miner

    global _cuda_miner_mod, _cuda_miner_fixed_launch_config_kernel
    cuda_miner.init(path)
    if autotune:
        _cuda_miner_fixed_launch_config_kernel = (
            cuda_miner.get_autotuned_cuda_miner_kernel(
                n_warmup_repeats=0, n_measure_repeats=2, show_progress=show_progress
            )
        )
    else:
        # TODO partial autotuning (grid size = number of SMs, autotune block size)
        # TODO find reliable default config: (512, 512) seems wrong
        _cuda_miner_fixed_launch_config_kernel = cuda_miner.FixedLaunchConfigKernel(
            cuda_miner.get_cuda_miner_kernel(),
            cuda_miner.CudaDeviceLaunchConfig((512,), (512,)),
        )
    _cuda_miner_mod = cuda_miner


_global_lock = threading.Lock()


class SideChannelItemType(enum.Enum):
    TRANSACTION_VALIDITY_PAIR = 0
    CHAIN_VALIDITY_PAIR = 1


def is_valid_block(block: "Block", valid_block_max_hash: int):
    if block is None:
        return True
    h = int.from_bytes(block.hash(), ENDIAN)
    return h <= valid_block_max_hash


def is_valid_chain(block: "Block", valid_block_max_hash: int):
    if block is None:
        return True
    while block.prev is not None:
        if not is_valid_block(block, valid_block_max_hash):
            return False
        block = block.prev
    return True


def block_is_ready(block: "Block"):
    if block is None:
        return True
    return block.nonce != 0


def chain_is_ready(chain: "Block"):
    """Returns True if all blocks in the chain have a nonce != 0. Note that 0 means not mined yet."""
    if chain is None:
        return True
    while chain.prev is not None:
        if not block_is_ready(chain):
            return False
        chain = chain.prev
    return True


def deserialize_private_key(key: bytes):
    p_bytes, q_bytes, d_bytes = (
        key[: KEY_SIZE // 16],
        key[KEY_SIZE // 16 : KEY_SIZE // 8],
        key[KEY_SIZE // 8 :],
    )
    p, q, d = (
        int.from_bytes(p_bytes, ENDIAN),
        int.from_bytes(q_bytes, ENDIAN),
        int.from_bytes(d_bytes, ENDIAN),
    )
    private_key = rsa.RSAPrivateNumbers(
        p=p,
        q=q,
        d=d,
        dmp1=d % (p - 1),
        dmq1=d % (q - 1),
        iqmp=rsa.rsa_crt_iqmp(p, q),
        public_numbers=rsa.RSAPublicNumbers(e=65537, n=p * q),
    ).private_key(default_backend())
    return private_key


def deserialize_public_key(key: bytes):
    public_numbers = int.from_bytes(key, ENDIAN)
    public_key = rsa.RSAPublicNumbers(e=65537, n=public_numbers).public_key(
        default_backend()
    )
    return public_key


def serialize_private_key(key):
    private_numbers = key.private_numbers()
    private_bytes = (
        private_numbers.p.to_bytes(KEY_SIZE // 16, ENDIAN)
        + private_numbers.q.to_bytes(KEY_SIZE // 16, ENDIAN)
        + private_numbers.d.to_bytes(KEY_SIZE // 8, ENDIAN)
    )
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


def serialize_balances(balances: Dict[bytes, int]) -> bytes:
    return b"".join(
        k + v.to_bytes(BALANCE_SERIALIZATION_VALUE_SIZE, ENDIAN)
        for k, v in balances.items()
    )


def deserialize_balances(raw: bytes) -> Dict[bytes, int]:
    return {
        raw[i : i + KEY_SIZE // 8]: int.from_bytes(
            raw[
                i + KEY_SIZE // 8 : i + KEY_SIZE // 8 + BALANCE_SERIALIZATION_VALUE_SIZE
            ],
            ENDIAN,
        )
        for i in range(0, len(raw), KEY_SIZE // 8 + BALANCE_SERIALIZATION_VALUE_SIZE)
    }


def execute_transaction(
    t: "Transaction",
    validator: bytes,
    balances: Dict[bytes, int],
    invalidated_uuids: set[bytes],
):
    balances[t.sender] = balances.get(t.sender, 0) - t.amount
    balances[t.receiver] = balances.get(t.receiver, 0) + t.amount - t.fee
    balances[validator] = balances.get(validator, 0) + t.fee
    invalidated_uuids.add(t.uuid)


def full_verify(
    block: "Block",
    init_balance: Dict[bytes, int],
    val_reward: int,
    const_transaction_fee: int,
    relative_transaction_fee: float,
    valid_block_max_hash: int,
):
    balances = init_balance.copy()
    invalidated_uuids = set()
    if block is None:
        return True, balances, invalidated_uuids
    if not is_valid_chain(block, valid_block_max_hash):
        return False, balances, invalidated_uuids

    while block is not None:
        for t in block.transactions:
            if not t.is_valid(
                balances,
                invalidated_uuids,
                const_transaction_fee,
                relative_transaction_fee,
            ):
                return False, balances, invalidated_uuids
            execute_transaction(t, block.validator, balances, invalidated_uuids)
        balances[block.validator] = balances.get(block.validator, 0) + val_reward
        block = block.prev
    return True, balances, invalidated_uuids


class Transaction:
    BYTE_COUNTS = [KEY_SIZE // 8, KEY_SIZE // 8, 8, 8, 8, KEY_SIZE // 8]
    N_BYTES = sum(BYTE_COUNTS)
    CONVERSION_F = [
        LXX,
        LXX,
        lambda b: int.from_bytes(b, ENDIAN),
        lambda b: int.from_bytes(b, ENDIAN),
        LXX,
        LXX,
    ]

    def __init__(
        self,
        sender: bytes,
        receiver: bytes,
        amount: int,
        fee: int,
        uuid: bytes,
        signature: bytes,
    ):
        self.sender = sender
        self.receiver = receiver
        self.amount = amount
        self.fee = fee
        self.uuid = uuid
        self.signature = signature

    def _bytes_without_signature(self):
        amount = self.amount.to_bytes(Transaction.BYTE_COUNTS[2], ENDIAN)
        transaction_fee = self.fee.to_bytes(Transaction.BYTE_COUNTS[3], ENDIAN)
        return self.sender + self.receiver + amount + transaction_fee + self.uuid

    def to_bytes(self):
        return self._bytes_without_signature() + self.signature

    def is_sane(self, const_transaction_fee: int, relative_transaction_fee: float):
        positive = self.amount > 0
        correct_fee = self.fee == const_transaction_fee + int(
            self.amount * relative_transaction_fee
        )
        amount_exceeds_fee = self.amount >= self.fee
        sender_not_receiver = self.sender != self.receiver
        signature_valid = verify_signature(
            self._bytes_without_signature(), self.signature, self.sender
        )
        conditions = (
            positive,
            correct_fee,
            amount_exceeds_fee,
            sender_not_receiver,
            signature_valid,
        )
        return all(conditions)

    def is_valid(
        self,
        balances: Dict[to_bytes, int],
        invalidated_uuids: set[to_bytes],
        const_transaction_fee: int,
        relative_transaction_fee: float,
        transaction_sane: bool = None,
    ):
        can_afford = balances.get(self.sender, 0) > self.amount > 0
        uuid_valid = self.uuid not in invalidated_uuids
        is_sane = (
            self.is_sane(const_transaction_fee, relative_transaction_fee)
            if transaction_sane is None
            else transaction_sane
        )
        conditions = (can_afford, uuid_valid, is_sane)
        return all(conditions)

    @classmethod
    def from_bytes(cls, raw: to_bytes):
        components = []
        cum_size = 0
        for size, cf in zip(Transaction.BYTE_COUNTS, Transaction.CONVERSION_F):
            components.append(cf(raw[cum_size : cum_size + size]))
            cum_size += size
        return cls(*components)


class Block:
    SIZE = 4
    BYTE_COUNTS = [NONCE_SIZE_BYTES, 32, 512, 4] + [Transaction.N_BYTES] * SIZE
    N_BYTES = sum(BYTE_COUNTS)
    CONVERSION_F = [
        lambda b: int.from_bytes(b, ENDIAN),
        LXX,
        LXX,
        lambda b: int.from_bytes(b, ENDIAN),
    ] + [Transaction.from_bytes] * SIZE

    def __init__(
        self,
        transactions: List["Transaction"],
        prev: "Block",
        validator: bytes,
        nonce: int = 0,
        version: int = 0,
    ):
        self.validator = validator
        self.transactions = transactions
        self.prev = prev
        self.nonce = nonce
        self.version = version
        self._cache_hash = None
        self._cache_inner_hash = None
        self._cache_val = nonce

    def _bytes_without_val(self):
        return (
            (
                self.prev.hash()
                if self.prev is not None
                else b"\0" * Block.BYTE_COUNTS[1]
            )
            + self.validator
            + self.version.to_bytes(Block.BYTE_COUNTS[3], ENDIAN)
            + b"".join(map(Transaction.to_bytes, self.transactions))
        )

    def to_bytes(self):
        return (
            self.nonce.to_bytes(Block.BYTE_COUNTS[0], ENDIAN)
            + self._bytes_without_val()
        )

    @classmethod
    def from_bytes(cls, raw: to_bytes):
        # NOTE: method assigns prev hash to prev member that should be Block object (gets linked externally)
        components = []
        cum_size = 0
        for size, cf in zip(Block.BYTE_COUNTS, Block.CONVERSION_F):
            components.append(cf(raw[cum_size : cum_size + size]))
            cum_size += size

        # last Block.SIZE components are transactions
        components, transactions = components[: -Block.SIZE], components[-Block.SIZE :]
        components.insert(0, transactions)
        transactions, nonce, prev, validator, version = components
        return cls(transactions, prev, validator, nonce, version)

    def _inner_hash(self):
        if self._cache_inner_hash is None:
            self._cache_inner_hash = sha256(self._bytes_without_val())
        return self._cache_inner_hash

    def hash(self):
        if self._cache_hash is None or self.nonce != self._cache_val:
            self._cache_hash = sha256(
                self.nonce.to_bytes(Block.BYTE_COUNTS[0], ENDIAN) + self._inner_hash()
            )
            self._cache_val = self.nonce
        return self._cache_hash


class NonceMiningJobHandler:
    def __init__(self):
        self.queued_jobs = []
        self._killed = threading.Event()
        self.worker = threading.Thread(target=self._worker, daemon=True)
        self.worker.start()

    def launch_job(
        self, block: "Block", use_cuda_miner: bool, valid_block_max_hash: int
    ):
        _global_lock.acquire()
        if any(b == block for b, _, _ in self.queued_jobs):
            _global_lock.release()
            self.kill()
            raise ValueError("a job has already been queued for this block")
        self.queued_jobs.append((block, use_cuda_miner, valid_block_max_hash))
        _global_lock.release()

    def kill(self):
        self._killed.set()
        self.worker.join()

    def __del__(self):
        self.kill()

    def _worker(self):
        while not self._killed.is_set():
            while not self.queued_jobs:
                if self._killed.is_set():
                    return
                thread_yield()
            _global_lock.acquire()
            job = self.queued_jobs.pop(0)
            _global_lock.release()
            block, use_cuda_miner, valid_block_max_hash = job
            if block.nonce != 0:
                raise ValueError(
                    "nonce already set to non-zero value, meaning it was already mined"
                )
            elif use_cuda_miner:
                if _cuda_miner_mod is None:
                    raise RuntimeError(
                        "CUDA miner not loaded; did you forget to call load_cuda_miner?"
                    )
                h = int.from_bytes(block._inner_hash(), ENDIAN)
                nonce_out = _cuda_miner_mod.mine(
                    _cuda_miner_fixed_launch_config_kernel,
                    h,
                    valid_block_max_hash,
                    self._killed,
                )
                if nonce_out is not None:
                    block.nonce = nonce_out
                    if not is_valid_block(block, valid_block_max_hash):
                        raise RuntimeError(
                            f"Miner produced invalid nonce {hex(nonce_out)}. Please report to developer."
                        )
            else:
                block_ = Block(
                    block.transactions, block.prev, block.validator, 1, block.version
                )
                while not self._killed.is_set() and not is_valid_block(
                    block_, valid_block_max_hash
                ):
                    block_.nonce += 1
                if not self._killed.is_set():
                    block.nonce = block_.nonce


def chain_from_bytes(raw: bytes):
    if raw == b"":
        return None
    block_bytes = [
        raw[i : i + Block.N_BYTES]
        for i in range(0, int(len(raw) / Block.N_BYTES) * Block.N_BYTES, Block.N_BYTES)
    ]
    blocks = list(map(Block.from_bytes, block_bytes))
    # link blocks
    for block in blocks:
        if block.prev == b"\0" * len(block.prev):
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
        blocks.remove(next_block)
        block = next_block
    return block


def chain_to_bytes(chain: "Block") -> bytes:
    out = b""
    while chain is not None:
        out += chain.to_bytes()
        chain = chain.prev
    return out


class OpenBlock:
    def __init__(
        self,
        prev: Optional["Block"],
        validator: bytes,
        transactions: List["Transaction"] = None,
    ):
        if transactions is None:
            transactions = []
        self.validator = validator
        self.transactions = transactions
        self.prev = prev

    def receive(
        self,
        transaction: "Transaction",
        balances: Dict[bytes, int],
        invalidated_uuids: set[bytes],
        const_transaction_fee: int,
        relative_transaction_fee: float,
        val_reward: int,
        transaction_sane: bool = None,
    ) -> tuple[bool, bool]:
        if not transaction.is_valid(
            balances,
            invalidated_uuids,
            const_transaction_fee,
            relative_transaction_fee,
            transaction_sane,
        ):
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
                if len(raw) % Block.N_BYTES != 0:
                    continue
                chain = chain_from_bytes(raw)
                if chain is not None:
                    out.append(chain)
                    n -= 1
    else:
        while n and chain_buffer:
            raw = chain_buffer.pop(0)
            if len(raw) % Block.N_BYTES != 0:
                continue
            chain = chain_from_bytes(raw)
            if chain is not None:
                out.append(chain)
                n -= 1
    return out


def broadcast_chain(broadcast_buffer, block: "Block"):
    if broadcast_buffer is None:
        return
    chain_bytes = chain_to_bytes(block)
    if isinstance(broadcast_buffer, (queue.Queue, mp.queues.Queue)):
        try:
            broadcast_buffer.put_nowait(chain_bytes)
        except queue.Full:
            broadcast_buffer.get_nowait()
            broadcast_buffer.put_nowait(chain_bytes)
    else:
        broadcast_buffer.append(chain_bytes)


def broadcast_balances(
    broadcast_buffer,
    balances: Dict[bytes, int],
    broadcast_copy: bool = False,
    broadcast_as_bytes: bool = False,
):
    if broadcast_buffer is None:
        return
    if broadcast_as_bytes:
        balances = serialize_balances(balances)
    elif broadcast_copy:
        balances = balances.copy()
    if isinstance(broadcast_buffer, (queue.Queue, mp.queues.Queue)):
        try:
            broadcast_buffer.put_nowait(balances)
        except queue.Full:
            broadcast_buffer.get_nowait()
            broadcast_buffer.put_nowait(balances)
    else:
        broadcast_buffer.append(balances)


def broadcast_side_channel(side_channel, item_type: SideChannelItemType, item):
    if side_channel is None:
        return
    if isinstance(side_channel, (queue.Queue, mp.queues.Queue)):
        try:
            side_channel.put_nowait((item_type, item))
        except queue.Full:
            side_channel.get_nowait()
            side_channel.put_nowait((item_type, item))
    else:
        side_channel.append((item_type, item))


def _mine(
    validator: bytes,
    chain: "Block" = None,
    transaction_recv_channel=None,
    chain_recv_channel=None,
    chain_broadcast_channel=None,
    balance_broadcast_channel=None,
    side_channel=None,
    init_balance=None,
    terminate_event=None,
    has_terminated_event=None,
    use_cuda_miner_event=None,
    incompatible_chain_distrust: int = 1,
    compatible_chain_distrust: int = 1,
    val_reward: int = 0,
    const_transaction_fee: int = 0,
    relative_transaction_fee: float = 0.0,
    max_recv_chains_per_iter: int = -1,
    max_recv_transactions_per_iter: int = -1,
    collect_time: int = 0,
    copy_balances_on_broadcast=False,
    transaction_backlog_size: int = 1,
    broadcast_balances_as_bytes: bool = False,
    valid_block_max_hash: int = 0x00FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF,
    version: int = 0,
):
    """
    The main blockchain function. It receives transactions and chains, verifies them, and updates the blockchain.
    :param validator: the public key of the miner
    :param chain: optional initial chain
    :param transaction_recv_channel: queue for receiving transactions
    :param chain_recv_channel: queue for receiving chains
    :param chain_broadcast_channel: queue for broadcasting new chains that a nonce was found for
    :param balance_broadcast_channel: queue for broadcasting balances for review
    :param side_channel: queue for broadcasting expensive to compute data that might be reused (eg chain validity)
    :param init_balance: initial balance of the blockchain (what the balance is before the first transaction)
    :param terminate_event: event set when the blockchain main loop should terminate
    :param has_terminated_event: event set when the blockchain main loop has terminated
    :param use_cuda_miner_event: event that signals that the miner should use the CUDA miner kernel instead of the debug miner
    :param incompatible_chain_distrust: how many blocks a chain that has incompatible balances with the current chain must be longer to replace the current chain
    :param compatible_chain_distrust: how many blocks a chain that has the same balances at the end has to be longer to replace the current chain
    :param val_reward: the reward for validating a block
    :param const_transaction_fee: the constant transaction fee that is subtracted from the amount sent and given to the validator
    :param relative_transaction_fee: percentage (value between 0 and 1) of the amount sent that is given to the validator
    :param max_recv_chains_per_iter: how many alternative chains may be processed in one loop iteration (helps limit time / iter). negative value means no limit
    :param max_recv_transactions_per_iter: how many transactions may be processed in one loop iteration (helps limit time / iter). negative value means no limit
    :param collect_time: how many nanoseconds to wait and collect transactions before processing them (allows waiting for more transactions to arrive so miner can select the most valuable ones)
    :param copy_balances_on_broadcast: whether to copy the balances dictionary when broadcasting it (helps prevent bugs by removing mutation coupling with blockchain core). Ignored if broadcast_balances_as_bytes is True.
    :param transaction_backlog_size: how many blocks' submitted transactions are stored in the backlog in case an alternative chain arrives (where those suddenly are valid)
    :param broadcast_balances_as_bytes: whether to broadcast balances as bytes or as a dictionary
    :param valid_block_max_hash: the max value of the hash (interpreted as integer) that makes a block valid
    :param version: the version of the blockchain (to allow for forks)
    """

    assert isinstance(version, int), "version must be integer"
    assert transaction_backlog_size >= 0, "transaction_backlog_size cannot be negative"
    assert max_recv_chains_per_iter != 0, "max_recv_chains_per_iter cannot be 0"
    assert (
        max_recv_transactions_per_iter != 0
    ), "max_recv_transactions_per_iter cannot be 0"
    assert collect_time >= 0, "collect time cannot be negative"
    assert (
        incompatible_chain_distrust > 0
    ), "an external chain cannot be trusted as much as or more than more than your own"
    assert (
        compatible_chain_distrust > 0
    ), "an external chain cannot be trusted as much as or more than more than your own"
    assert val_reward >= 0, "you cannot have a negative nonce reward"
    assert const_transaction_fee >= 0, "you cannot have a negative transaction fee"
    assert (
        0 <= relative_transaction_fee < 1
    ), "relative transaction fee must be in [0, 1)"
    assert len(validator) == KEY_SIZE // 8, "invalid validator public key size"
    assert (
        chain is None
        or full_verify(
            chain,
            init_balance,
            val_reward,
            const_transaction_fee,
            relative_transaction_fee,
            valid_block_max_hash,
        )[0]
    ), "invalid initial chain"
    assert transaction_recv_channel is None or isinstance(
        transaction_recv_channel, (queue.Queue, mp.queues.Queue, list)
    )
    assert chain_recv_channel is None or isinstance(
        chain_recv_channel, (queue.Queue, mp.queues.Queue, list)
    )
    assert chain_broadcast_channel is None or isinstance(
        chain_broadcast_channel, (queue.Queue, mp.queues.Queue, list)
    )
    assert balance_broadcast_channel is None or isinstance(
        balance_broadcast_channel, (queue.Queue, mp.queues.Queue, list)
    )
    assert side_channel is None or isinstance(
        side_channel, (queue.Queue, mp.queues.Queue, list)
    )
    assert init_balance is None or isinstance(init_balance, dict)
    assert terminate_event is None or isinstance(
        terminate_event, (threading.Event, mp.synchronize.Event)
    )
    assert has_terminated_event is None or isinstance(
        has_terminated_event, (threading.Event, mp.synchronize.Event)
    )
    assert use_cuda_miner_event is None or isinstance(
        use_cuda_miner_event, (threading.Event, mp.synchronize.Event)
    )
    assert isinstance(incompatible_chain_distrust, int)
    assert isinstance(compatible_chain_distrust, int)
    assert isinstance(val_reward, int)
    assert isinstance(const_transaction_fee, int)
    assert isinstance(relative_transaction_fee, float)
    assert isinstance(max_recv_chains_per_iter, int)
    assert isinstance(max_recv_transactions_per_iter, int)
    assert isinstance(collect_time, int)
    assert isinstance(copy_balances_on_broadcast, bool)
    assert isinstance(transaction_backlog_size, int)
    assert isinstance(broadcast_balances_as_bytes, bool)
    assert isinstance(validator, bytes)
    assert isinstance(chain, Block) or chain is None
    assert isinstance(init_balance, dict) or init_balance is None

    if RUN_WEAKNESS_DETECTION_WARNINGS:
        if val_reward != 0:
            print(HAS_VAL_REWARD_WARNING)

    if init_balance is None:
        init_balance = {}
    open_block, balances, invalidated_uuids = (
        OpenBlock(chain, validator),
        init_balance,
        set(),
    )
    transaction_backlog = [[]] * transaction_backlog_size
    mining_job_handler = NonceMiningJobHandler()
    updated_chain = False
    replaced_chain = False
    last_loop_time_ns = 0

    while terminate_event is None or not terminate_event.is_set():
        while time.time_ns() - last_loop_time_ns < collect_time and (
            terminate_event is None or not terminate_event.is_set()
        ):
            thread_yield()
        last_loop_time_ns = time.time_ns()
        current_chain_len = chain_len(chain)

        # receive new chains
        recv_chains = get_received_chains(chain_recv_channel, max_recv_chains_per_iter)
        favorite_new_chain_data = (None, None, None)
        max_trust_level = 0
        for recv_chain in recv_chains:
            new_chain_len = chain_len(recv_chain)
            is_valid, new_balances, new_invalidated_uuids = full_verify(
                recv_chain,
                init_balance,
                val_reward,
                const_transaction_fee,
                relative_transaction_fee,
                valid_block_max_hash,
            )
            if side_channel is not None:
                broadcast_side_channel(
                    side_channel,
                    SideChannelItemType.CHAIN_VALIDITY_PAIR,
                    (recv_chain.to_bytes(), is_valid),
                )
            # if chains are equivalent, trust them more than if the result is different
            if is_valid and (
                new_chain_len - current_chain_len - incompatible_chain_distrust + 1
                > max_trust_level
                or (
                    new_chain_len - current_chain_len - compatible_chain_distrust + 1
                    > max_trust_level
                    and new_balances == balances
                )
            ):
                favorite_new_chain_data = (
                    recv_chain,
                    new_balances,
                    new_invalidated_uuids,
                )
                max_trust_level = max(
                    new_chain_len - current_chain_len - incompatible_chain_distrust + 1,
                    new_chain_len - current_chain_len - compatible_chain_distrust + 1,
                )
        if favorite_new_chain_data[0] is not None:
            chain, balances, invalidated_uuids = favorite_new_chain_data
            mining_job_handler.kill()
            mining_job_handler = NonceMiningJobHandler()
            open_block = OpenBlock(chain, validator)
            replaced_chain = True
            updated_chain = True

        # receive new transactions
        recv_trans = get_received_transactions(
            transaction_recv_channel, max_recv_transactions_per_iter
        )
        t_is_sane = [
            t.is_sane(const_transaction_fee, relative_transaction_fee)
            for t in recv_trans
        ]
        if side_channel is not None:
            for t, sane in zip(recv_trans, t_is_sane):
                if not sane:
                    broadcast_side_channel(
                        side_channel,
                        SideChannelItemType.TRANSACTION_VALIDITY_PAIR,
                        (t.to_bytes(), False),
                    )
        filtered_recv_transactions = [
            t for t, sane in zip(recv_trans, t_is_sane) if sane
        ]
        recv_sane_transactions = sorted(
            filtered_recv_transactions, key=lambda t: t.amount, reverse=True
        )
        transactions = recv_sane_transactions
        if replaced_chain:
            transactions = sum(transaction_backlog, start=[]) + transactions
        for t in transactions:
            if transaction_backlog_size:
                transaction_backlog[-1].append(t)
            is_closed, is_valid_t = open_block.receive(
                t,
                balances,
                invalidated_uuids,
                const_transaction_fee,
                relative_transaction_fee,
                val_reward,
                True,
            )
            if side_channel is not None:
                broadcast_side_channel(
                    side_channel,
                    SideChannelItemType.TRANSACTION_VALIDITY_PAIR,
                    (t.to_bytes(), is_valid_t),
                )
            if is_closed:
                # NOTE: sorting makes miner network more resilient against potential problems caused by blocks being
                # received in different orders by different miners
                sorted_transactions = sorted(
                    open_block.transactions, key=lambda t: t.amount, reverse=True
                )
                chain = Block(sorted_transactions, chain, validator, version=version)
                mining_job_handler.launch_job(
                    chain,
                    use_cuda_miner_event is not None and use_cuda_miner_event.is_set(),
                    valid_block_max_hash,
                )
                open_block = OpenBlock(chain, validator)
                # remove those that were just closed
                transaction_backlog.pop(0)
                transaction_backlog.append([])
                updated_chain = True

        if updated_chain and chain_is_ready(chain):
            broadcast_chain(chain_broadcast_channel, chain)
            broadcast_balances(
                balance_broadcast_channel,
                balances,
                copy_balances_on_broadcast,
                broadcast_balances_as_bytes,
            )
            updated_chain = False

    mining_job_handler.kill()
    if has_terminated_event is not None:
        has_terminated_event.set()


class MiningConfig:
    def __init__(
        self,
        incompatible_chain_distrust: int = 1,
        compatible_chain_distrust: int = 1,
        val_reward: int = 0,
        const_transaction_fee: int = 0,
        relative_transaction_fee: float = 0.0,
        max_recv_chains_per_iter: int = 1,
        max_recv_transactions_per_iter: int = 4,
        collect_time: int = 0,
        copy_balances_on_broadcast: bool = True,
        transaction_backlog_size: int = 2,
        broadcast_balances_as_bytes: bool = True,
        valid_block_max_hash: int = 0x00FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF,
        version: int = 0,
    ):
        self.incompatible_chain_distrust = incompatible_chain_distrust
        self.compatible_chain_distrust = compatible_chain_distrust
        self.val_reward = val_reward
        self.const_transaction_fee = const_transaction_fee
        self.relative_transaction_fee = relative_transaction_fee
        self.max_recv_chains_per_iter = max_recv_chains_per_iter
        self.max_recv_transactions_per_iter = max_recv_transactions_per_iter
        self.collect_time = collect_time
        self.copy_balances_on_broadcast = copy_balances_on_broadcast
        self.transaction_backlog_size = transaction_backlog_size
        self.broadcast_balances_as_bytes = broadcast_balances_as_bytes
        self.valid_block_max_hash = valid_block_max_hash
        self.version = version


class Miner:
    """
    A class that represents a miner. It is a wrapper around the _mine function that allows for easy
    serialization and deserialization of the blockchain. You can add transactions and alternative chains to the blockchain.
    It allows for easy retrieval of the balances from the blockchain.
    """

    def __init__(
        self,
        validator: bytes,
        chain: "Block" = None,
        init_balance=None,
        config: MiningConfig = None,
        run_with: str = "direct",
    ):
        super().__init__()
        self.validator = validator
        self.chain = chain
        self.init_balance = init_balance.copy() if init_balance is not None else {}
        if config is None:
            config = MiningConfig()

        if run_with == "threading":
            self.transaction_recv_buffer = queue.Queue()
            self.chain_recv_buffer = queue.Queue()
            self.chain_broadcast_buffer = queue.Queue(1)
            self.balance_broadcast_buffer = queue.Queue(1)
            self.side_channel = queue.Queue()
            self.terminate_event = threading.Event()
            self.has_terminated_event = threading.Event()
            self.use_cuda_miner_event = threading.Event()
        elif run_with == "mp":
            self.transaction_recv_buffer = mp.Queue()
            self.chain_recv_buffer = mp.Queue()
            self.chain_broadcast_buffer = mp.Queue(1)
            self.balance_broadcast_buffer = mp.Queue(1)
            self.side_channel = mp.Queue()
            self.terminate_event = mp.Event()
            self.has_terminated_event = mp.Event()
            self.use_cuda_miner_event = mp.Event()
        else:
            self.transaction_recv_buffer = queue.Queue()
            self.chain_recv_buffer = queue.Queue()
            self.chain_broadcast_buffer = queue.Queue(1)
            self.balance_broadcast_buffer = queue.Queue(1)
            self.side_channel = queue.Queue()
            self.terminate_event = threading.Event()
            self.use_cuda_miner_event = threading.Event()

        self._chain_queue_offload_queue = []
        self._transaction_queue_offload_queue = []

        self.incompatible_chain_distrust = config.incompatible_chain_distrust
        self.compatible_chain_distrust = config.compatible_chain_distrust
        self.val_reward = config.val_reward
        self.const_transaction_fee = config.const_transaction_fee
        self.relative_transaction_fee = config.relative_transaction_fee
        self.max_recv_chains_per_iter = config.max_recv_chains_per_iter
        self.max_recv_transactions_per_iter = config.max_recv_transactions_per_iter
        self.collect_time = config.collect_time
        self.copy_balances_on_broadcast = config.copy_balances_on_broadcast
        self.transaction_backlog_size = config.transaction_backlog_size
        self.broadcast_balances_as_bytes = config.broadcast_balances_as_bytes
        self.valid_block_max_hash = config.valid_block_max_hash
        self.version = config.version

        self.run_with = run_with
        self._get_balances_last = self.init_balance
        self._get_chain_last = self.chain

        args = (
            self.validator,
            self.chain,
            self.transaction_recv_buffer,
            self.chain_recv_buffer,
            self.chain_broadcast_buffer,
            self.balance_broadcast_buffer,
            self.side_channel,
            self.init_balance,
            self.terminate_event,
            self.has_terminated_event,
            self.use_cuda_miner_event,
            self.incompatible_chain_distrust,
            self.compatible_chain_distrust,
            self.val_reward,
            self.const_transaction_fee,
            self.relative_transaction_fee,
            self.max_recv_chains_per_iter,
            self.max_recv_transactions_per_iter,
            self.collect_time,
            self.copy_balances_on_broadcast,
            self.transaction_backlog_size,
            self.broadcast_balances_as_bytes,
            self.valid_block_max_hash,
            self.version,
        )
        if run_with == "direct":
            _mine(*args)
        elif run_with == "threading":
            self._thread = threading.Thread(target=_mine, args=args)
            self._thread.start()
        elif run_with == "mp":
            self._process = mp.Process(target=_mine, args=args)
            self._process.start()
        else:
            raise ValueError(
                "invalid run_with value: run_with must be one of ('direct', 'threading', 'mp')"
            )

    def to_bytes(self):
        return self.validator + chain_to_bytes(self.chain)

    @classmethod
    def from_bytes(cls, raw: bytes, **kwargs):
        validator, chain_bytes = raw[: KEY_SIZE // 8], raw[KEY_SIZE // 8 :]
        chain = chain_from_bytes(chain_bytes)
        return cls(validator, chain, **kwargs)

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
        if (
            self.run_with == "mp"
            and block_until_terminated
            and self._process.is_alive()
        ):
            self._process.kill()  # .join() for some reason hangs
        elif (
            self.run_with == "threading"
            and block_until_terminated
            and self._thread.is_alive()
        ):
            self._thread.join()

    def finish(
        self,
    ) -> tuple[Optional[bytes], Optional[Union[Dict[bytes, int], bytes]]]:
        final_chain, final_balances = self.get_chain(), self.get_balances()
        self.kill(True)
        return final_chain, final_balances

    @staticmethod
    def _safe_add_to_queue(item, q, offload_queue):
        try:
            if offload_queue:
                q.put_nowait(offload_queue.pop(0))
            else:
                q.put_nowait(item)
        except queue.Full:
            offload_queue.append(item)

    def add_transaction(self, transaction: bytes):
        self._safe_add_to_queue(
            transaction,
            self.transaction_recv_buffer,
            self._transaction_queue_offload_queue,
        )

    def add_chain(self, chain: bytes):
        self._safe_add_to_queue(
            chain, self.chain_recv_buffer, self._chain_queue_offload_queue
        )

    def get_balances(self) -> Optional[Union[bytes, Dict[bytes, int]]]:
        out = None
        while not self.balance_broadcast_buffer.empty():
            out = self.balance_broadcast_buffer.get_nowait()

        if out is not None:
            self._get_balances_last = out
        return self._get_balances_last

    def get_chain(self) -> Optional[bytes]:
        out = None
        while not self.chain_broadcast_buffer.empty():
            out = self.chain_broadcast_buffer.get_nowait()

        if out is not None:
            self._get_chain_last = out
        return self._get_chain_last

    def get_side_channel_item(self) -> Optional[tuple[SideChannelItemType, Any]]:
        if not self.side_channel.empty():
            return self.side_channel.get_nowait()

    def use_cuda_miner(self):
        self.use_cuda_miner_event.set()

    def use_debug_miner(self):
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
