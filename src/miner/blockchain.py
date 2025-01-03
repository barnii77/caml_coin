import threading
import math
import multiprocessing as mp
import queue
import enum
import time
import random
import cupy as cp

from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.asymmetric.utils import (
    decode_dss_signature,
    encode_dss_signature,
)
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.backends import default_backend
from cryptography.exceptions import InvalidSignature
from typing import Optional, Union, Any, List, Dict, Callable, Set

from miner_utils import thread_yield, sha256, Shared, LinkedEvent


# fix python std lib bug in mp (have to create queue before I can access mp.queues and same with event and mp.synchronize)
try:
    mp.Queue()
    mp.Event()
except AttributeError:
    raise RuntimeError(
        "mp.queues or mp.synchronize submodule could not be loaded. This is a weirdness in python stdlib... sry :("
    )

_cuda_miner_mod = None
_cuda_miner_fixed_launch_config_kernel = None
_cuda_miner_load_callback_args = None

# noli me tangere (do not touch)
NONCE_SIZE_BYTES = 16
RUN_WEAKNESS_DETECTION_WARNINGS = True
ENDIAN = "little"
PUB_KEY_DECODE_ENDIAN = "big"
LXX = lambda x: x
BALANCE_SERIALIZATION_VALUE_SIZE = 16
PUBLIC_KEY_SIZE = 264
HAS_VAL_REWARD_WARNING = """Warning, val_reward > 0.
Make sure miners can't do a self-reward attack.
A self-reward attack is when a miner spams transactions from one of his accounts to another (without broadcasting) and mines on those transactions.
This way he can get the reward without actually doing anything (useful)."""  # NOTE: if self-reward "attack" is the default behaviour and everyone does it, it's not really an attack lol
# FIXME fee_density seems broken and should currently not be used
# TODO basic must not be used in production because it encourages dropping low-profit transactions
TRANSACTIONS_SORT_MODE = "basic"  # fee_density / basic


def schedule_jit_cuda_miner_load(*args):
    global _cuda_miner_load_callback_args
    _cuda_miner_load_callback_args = args


def load_cuda_miner(
    path="cuda_miner_templates/src/nonce128_emulated_uint128_random_anywhere.cu",
    n_blocks_if_properties_unavailable=14,
    n_threads_per_block_if_properties_unavailable=512,
    device_ids=None,
):
    # autotuner doesn't work
    autotune = False
    show_progress = False
    if device_ids is None:
        device_ids = [0]
    import cuda_miner

    global _cuda_miner_mod, _cuda_miner_fixed_launch_config_kernel, _cuda_miner_load_callback_args
    _cuda_miner_mod = cuda_miner
    _cuda_miner_load_callback_args = (
        path,
        n_blocks_if_properties_unavailable,
        n_threads_per_block_if_properties_unavailable,
        device_ids,
    )
    cuda_miner.init(path)
    if autotune:
        _cuda_miner_fixed_launch_config_kernel = (
            cuda_miner.get_autotuned_cuda_miner_kernel(device_ids, 0, 1, show_progress)
        )
    else:
        configs = {}
        for device_id in device_ids:
            device_properties = cp.cuda.runtime.getDeviceProperties(device_id)
            max_multiprocessors = device_properties.get(
                "multiProcessorCount", n_blocks_if_properties_unavailable
            )
            max_threads_per_block = device_properties.get(
                "maxThreadsPerBlock", n_threads_per_block_if_properties_unavailable
            )
            configs[device_id] = cuda_miner.CudaDeviceLaunchConfig(
                (max_multiprocessors,), (max_threads_per_block,)
            )
        config = cuda_miner.CudaMultiDeviceLaunchConfig(configs)
        _cuda_miner_fixed_launch_config_kernel = cuda_miner.FixedLaunchConfigKernel(
            cuda_miner.get_cuda_miner_kernel(),
            config,
        )


_global_lock = threading.Lock()


class SideChannelItemType(enum.Enum):
    # NOTE: these values have to be 0, 1, 2 because they are used as indices in app_miner.py
    TRANSACTION_FEEDBACK_PAIR = 0
    CHAIN_FEEDBACK_PAIR = 1
    BLOCK_FEEDBACK_PAIR = 2


class FeedbackCode(enum.Enum):
    TRANSACTION_INSANE = enum.auto()
    TRANSACTION_REJECTED_BUT_SANE = enum.auto()
    TRANSACTION_ACCEPTED = enum.auto()
    CHAIN_INVALID = enum.auto()
    CHAIN_VALID_BUT_NOT_ACCEPTED = enum.auto()
    CHAIN_ACCEPTED = enum.auto()
    BLOCK_REJECTED = enum.auto()
    BLOCK_ACCEPTED = enum.auto()


def format_feedback_code(code: "FeedbackCode"):
    if code == FeedbackCode.TRANSACTION_ACCEPTED:
        return "TransactionAccepted"
    elif code == FeedbackCode.TRANSACTION_INSANE:
        return "TransactionInsane"
    elif code == FeedbackCode.TRANSACTION_REJECTED_BUT_SANE:
        return "TransactionRejectedButSane"
    elif code == FeedbackCode.CHAIN_INVALID:
        return "ChainInvalid"
    elif code == FeedbackCode.CHAIN_VALID_BUT_NOT_ACCEPTED:
        return "ChainValidButNotAccepted"
    elif code == FeedbackCode.CHAIN_ACCEPTED:
        return "ChainAccepted"
    elif code == FeedbackCode.BLOCK_REJECTED:
        return "BlockRejected"
    elif code == FeedbackCode.BLOCK_ACCEPTED:
        return "BlockAccepted"
    else:
        return "InvalidFeedbackCode"


def is_accepted_feedback(code: "FeedbackCode"):
    return code in (
        FeedbackCode.CHAIN_ACCEPTED,
        FeedbackCode.BLOCK_ACCEPTED,
        FeedbackCode.TRANSACTION_ACCEPTED,
    )


class TransactionSortingTreeNode:
    def __init__(
        self,
        transaction: "Transaction",
        direct_dependencies: Set["TransactionSortingTreeNode"],
    ):
        self.transaction = transaction
        self.direct_dependencies = direct_dependencies
        self._recursive_dependencies: Set["TransactionSortingTreeNode"] = None

    def recursive_deps(self) -> Set["TransactionSortingTreeNode"]:
        if self._recursive_dependencies is None:
            indirect_deps = set()
            for dd in self.direct_dependencies:
                indirect_deps.update(dd.recursive_deps())
            self._recursive_dependencies = indirect_deps
        return self._recursive_dependencies


def select_top_transactions(transactions: Set["Transaction"], block_size: int):
    # build a dependency graph for the transactions
    t_by_timestamp = sorted(
        transactions, key=lambda t: int.from_bytes(t.timestamp, ENDIAN)
    )
    if TRANSACTIONS_SORT_MODE == "basic":
        return t_by_timestamp[:block_size]

    incoming = {}
    tree_nodes = set()
    for t in t_by_timestamp:
        tn = TransactionSortingTreeNode(t, incoming.get(t.sender, set()).copy())
        tree_nodes.add(tn)
        if t.receiver not in incoming:
            incoming[t.receiver] = set()
        incoming[t.receiver].add(tn)

    # resolve the recursive dependencies of each node
    tn_to_rec_deps: Dict[
        "TransactionSortingTreeNode", Set["TransactionSortingTreeNode"]
    ] = {}
    for tn in tree_nodes:
        tn_to_rec_deps[tn] = tn.recursive_deps()

    transactions_to_mine = set()
    while len(transactions_to_mine) < block_size:
        # compute total fee of all deps + node and number of nodes to compute density and filter out unreachable transactions later
        fee_density_info: Dict["TransactionSortingTreeNode", tuple[int, int]] = (
            {}
        )  # tuple[cumulative_fee, n_nodes]
        for tn in tree_nodes:
            rec_deps = tn_to_rec_deps[tn]
            total_fee = sum(
                map(lambda tn: tn.transaction.fee, rec_deps), start=tn.transaction.fee
            )
            fee_density_info[tn] = (total_fee, len(rec_deps) + 1)

        fee_densities = {
            tn: (total_fee / n_nodes)
            for tn, (total_fee, n_nodes) in fee_density_info.items()
            if n_nodes <= (block_size - len(transactions_to_mine))
        }
        max_density_tn = max(fee_densities, key=lambda k: fee_densities[k])

        update = set(max_density_tn.recursive_deps()) | {max_density_tn}
        transactions_to_mine.update(map(lambda tn: tn.transaction, update))

        # patch the tn_to_rec_deps dict to not include these transactions as keys and remove them from all deps sets
        tree_nodes.difference_update(update)
        for tn in update:
            tn_to_rec_deps.pop(tn)
        for v in tn_to_rec_deps.values():
            v.difference_update(update)

    return sorted(
        transactions_to_mine, key=lambda t: int.from_bytes(t.timestamp, ENDIAN)
    )


def block_list(chain: "Block") -> List["Block"]:
    """Returns a list of the blocks in a blockchain from earliest to latest"""
    out = []
    while chain is not None:
        out.append(chain)
        chain = chain.prev
    out.reverse()
    return out


def reversed_block_list(chain: "Block") -> List["Block"]:
    """Returns a list of the blocks in a blockchain from latest to earliest"""
    out = []
    while chain is not None:
        out.append(chain)
        chain = chain.prev
    return out


def is_valid_block(block: "Block", valid_block_max_hash: int):
    return block is None or int.from_bytes(block.hash(), ENDIAN) <= valid_block_max_hash


def is_valid_chain(block: "Block", valid_block_max_hash: int):
    while block is not None:
        if not is_valid_block(block, valid_block_max_hash):
            return False
        block = block.prev
    return True


def block_is_ready(block: "Block"):
    if block is None:
        return True
    return block.nonce != 0


def deserialize_private_key(key: bytes):
    private_value = int.from_bytes(key, ENDIAN)
    private_key = ec.derive_private_key(
        private_value, ec.SECP256R1(), default_backend()
    )
    return private_key


def deserialize_public_key(key: bytes):
    prefix, x = key[:1], key[1:]
    if prefix not in (b"\x02", b"\x03"):
        raise ValueError("invalid public key prefix")
    x_decode_endian = x if ENDIAN == PUB_KEY_DECODE_ENDIAN else x[::-1]
    return ec.EllipticCurvePublicKey.from_encoded_point(
        ec.SECP256R1(), prefix + x_decode_endian
    )


def serialize_private_key(key):
    private_value = key.private_numbers().private_value
    private_bytes = private_value.to_bytes(PUBLIC_KEY_SIZE // 8 - 1, ENDIAN)
    return private_bytes


def serialize_public_key(key):
    public_numbers = key.public_numbers()
    x_bytes = public_numbers.x.to_bytes(PUBLIC_KEY_SIZE // 8 - 1, ENDIAN)
    prefix = b"\x02" if public_numbers.y % 2 == 0 else b"\x03"
    public_bytes = prefix + x_bytes
    return public_bytes


def sign_message(message: bytes, private_key_bytes: bytes):
    private_key = deserialize_private_key(private_key_bytes)
    der_signature = private_key.sign(message, ec.ECDSA(hashes.SHA256()))
    r, s = decode_dss_signature(der_signature)
    r_bytes = r.to_bytes(PUBLIC_KEY_SIZE // 8 - 1, ENDIAN)
    s_bytes = s.to_bytes(PUBLIC_KEY_SIZE // 8 - 1, ENDIAN)
    return r_bytes + s_bytes


def verify_signature(message: bytes, signature: bytes, public_key_bytes: bytes):
    try:
        public_key = deserialize_public_key(public_key_bytes)
    except ValueError:
        return False
    r = int.from_bytes(signature[: PUBLIC_KEY_SIZE // 8 - 1], ENDIAN)
    s = int.from_bytes(signature[PUBLIC_KEY_SIZE // 8 - 1 :], ENDIAN)
    raw_signature = encode_dss_signature(r, s)
    try:
        public_key.verify(raw_signature, message, ec.ECDSA(hashes.SHA256()))
    except InvalidSignature:
        return False
    return True


def gen_key_pair() -> tuple[bytes, bytes]:
    private_key = ec.generate_private_key(ec.SECP256R1(), default_backend())
    private_bytes = serialize_private_key(private_key)
    public_key = private_key.public_key()
    public_bytes = serialize_public_key(public_key)

    return private_bytes, public_bytes


def chain_len(block: "Block"):
    x = 0
    while block is not None:
        x += 1
        block = block.prev
    return x


def get_n_unready_blocks(chain: "Block"):
    i = 0
    while not block_is_ready(chain):
        i += 1
        chain = chain.prev
    return i


def verified_subchain(chain: "Block") -> Optional["Block"]:
    """A chain can potentially start with n blocks that are not verified yet.
    This function will split the unverified part off and return the verified part."""
    while chain is not None:
        if block_is_ready(chain):
            return chain
        chain = chain.prev
    return None


def serialize_balances(balances: Dict[bytes, int]) -> bytes:
    return b"".join(
        k + v.to_bytes(BALANCE_SERIALIZATION_VALUE_SIZE, ENDIAN)
        for k, v in balances.items()
    )


def deserialize_balances(raw: bytes) -> Dict[bytes, int]:
    return {
        raw[i : i + PUBLIC_KEY_SIZE // 8]: int.from_bytes(
            raw[
                i
                + PUBLIC_KEY_SIZE // 8 : i
                + PUBLIC_KEY_SIZE // 8
                + BALANCE_SERIALIZATION_VALUE_SIZE
            ],
            ENDIAN,
        )
        for i in range(
            0, len(raw), PUBLIC_KEY_SIZE // 8 + BALANCE_SERIALIZATION_VALUE_SIZE
        )
    }


def execute_transaction(
    t: "Transaction",
    validator: bytes,
    balances: Dict[bytes, int],
    invalidated_timestamps: Set[bytes],
):
    balances[t.sender] = balances.get(t.sender, 0) - t.amount - t.fee
    balances[t.receiver] = balances.get(t.receiver, 0) + t.amount
    balances[validator] = balances.get(validator, 0) + t.fee
    invalidated_timestamps.add(t.timestamp)


def undo_transaction(
    t: "Transaction",
    validator: bytes,
    balances: Dict[bytes, int],
    invalidated_timestamps: Set[bytes],
):
    if balances.get(t.receiver, 0) < t.amount:
        raise RuntimeError(
            "Receiver has too little money to send back. This is probably a bug. Please report to developer."
        )
    if balances.get(validator, 0) < t.fee:
        raise RuntimeError(
            "Receiver has too little money to send back. This is probably a bug. Please report to developer."
        )
    balances[t.sender] = balances.get(t.sender, 0) + t.amount + t.fee
    balances[t.receiver] = balances.get(t.receiver, 0) - t.amount
    balances[validator] = balances.get(validator, 0) - t.fee
    if t.timestamp in invalidated_timestamps:
        invalidated_timestamps.remove(t.timestamp)


def full_verify(
    chain: "Block",
    init_balances: Dict[bytes, int],
    val_reward: int,
    const_transaction_fee: int,
    relative_transaction_fee: float,
    valid_block_max_hash: int,
    max_timestamp_now_diff: int,
    *,
    _depth=0,
) -> tuple[bool, dict, set]:
    """
    Full verification run of a chain.
    If _depth > 0, only verify the last _depth blocks, all blocks otherwise.
    init_balances needs to be the balances at verification starting point.
    With 0, that's the true init, with bigger than zero, it's balances after chain_len - n blocks.
    """
    balances = init_balances.copy()
    min_timestamp_values = []
    min_ts_running_min = 1 << 64
    # FIXME I think fee density sorting breaks this logic
    # TODO figure out a way to filter the invalidated_timestamps for it to not grow too big
    if TRANSACTIONS_SORT_MODE == "basic":
        for block in reversed_block_list(chain)[: _depth if _depth > 0 else None]:
            # NOTE: the second arg is not actually guaranteed to be less than min_ts_running_min because of fee density sorting
            min_ts_running_min = min(
                min_ts_running_min,
                min(
                    map(
                        lambda transaction: int.from_bytes(
                            transaction.timestamp, ENDIAN
                        ),
                        block.transactions,
                    )
                ),
            )
            min_timestamp_values.insert(0, min_ts_running_min)

    invalidated_timestamps = set()
    if chain is None:
        return True, balances, invalidated_timestamps
    if not is_valid_chain(chain, valid_block_max_hash):
        return False, balances, invalidated_timestamps

    for i, block in enumerate(block_list(chain)[-max(_depth, 0) :]):
        if len(block.transactions) != Block.SIZE:
            return False, balances, invalidated_timestamps
        for t in block.transactions:
            if not t.is_valid(
                balances,
                invalidated_timestamps,
                const_transaction_fee,
                relative_transaction_fee,
                None,
                max_timestamp_now_diff,
            ):
                return False, balances, invalidated_timestamps
            execute_transaction(t, block.validator, balances, invalidated_timestamps)
        balances[block.validator] = balances.get(block.validator, 0) + val_reward

        # filter invalidated timestamps if in basic mode, that is, transactions are processed in exact timestamp order.
        # in that case, one can assume timestamps to be strictly monotonously growing and can therefore throw some out.
        if TRANSACTIONS_SORT_MODE == "basic":
            invalidated_timestamps = set(
                filter(
                    lambda timestamp: int.from_bytes(timestamp, ENDIAN)
                    >= min_timestamp_values[i],
                    invalidated_timestamps,
                )
            )
    return True, balances, invalidated_timestamps


class Transaction:
    BYTE_COUNTS = [
        PUBLIC_KEY_SIZE // 8,
        PUBLIC_KEY_SIZE // 8,
        8,
        8,
        8,
        8,
        2 * (PUBLIC_KEY_SIZE // 8 - 1),
    ]
    N_BYTES = sum(BYTE_COUNTS)
    CONVERSION_F = [
        LXX,
        LXX,
        lambda b: int.from_bytes(b, ENDIAN),
        lambda b: int.from_bytes(b, ENDIAN),
        LXX,
        LXX,
        LXX,
    ]

    def __init__(
        self,
        sender: bytes,
        receiver: bytes,
        amount: int,
        fee: int,
        timestamp: bytes,
        recv_time: bytes,
        signature: bytes,
    ):
        self.sender = sender
        self.receiver = receiver
        self.amount = amount
        self.fee = fee
        self.timestamp = timestamp
        self.recv_time = recv_time
        self.signature = signature
        self._is_padding = False

    @classmethod
    def create(
        cls,
        sender: bytes,
        receiver: bytes,
        amount: int,
        fee: int,
        recv_time: bytes,
        private_key: bytes,
    ):
        timestamp_bytes = time.time_ns().to_bytes(Transaction.BYTE_COUNTS[4], ENDIAN)
        signature = sign_message(
            sender
            + receiver
            + amount.to_bytes(Transaction.BYTE_COUNTS[2], ENDIAN)
            + fee.to_bytes(Transaction.BYTE_COUNTS[3], ENDIAN)
            + timestamp_bytes,
            private_key,
        )
        return cls(sender, receiver, amount, fee, timestamp_bytes, recv_time, signature)

    def verify_signature_input(self):
        amount = self.amount.to_bytes(Transaction.BYTE_COUNTS[2], ENDIAN)
        transaction_fee = self.fee.to_bytes(Transaction.BYTE_COUNTS[3], ENDIAN)
        return self.sender + self.receiver + amount + transaction_fee + self.timestamp

    def to_bytes(self):
        amount = self.amount.to_bytes(Transaction.BYTE_COUNTS[2], ENDIAN)
        transaction_fee = self.fee.to_bytes(Transaction.BYTE_COUNTS[3], ENDIAN)
        return (
            self.sender
            + self.receiver
            + amount
            + transaction_fee
            + self.timestamp
            + self.recv_time
            + self.signature
        )

    def is_sane(self, const_transaction_fee: int, relative_transaction_fee: float):
        positive = self.amount >= 0 and self.fee >= 0
        valid_fee = self.fee >= const_transaction_fee + math.ceil(  # ceil for more money ;)
            self.amount * relative_transaction_fee
        )
        sender_not_receiver = self.sender != self.receiver

        # check above conditions now because signature verification is expensive
        conditions = (positive, valid_fee, sender_not_receiver)
        if not all(conditions):
            return False
        signature_valid = verify_signature(
            self.verify_signature_input(), self.signature, self.sender
        )
        return signature_valid

    def is_valid(
        self,
        balances: Dict[bytes, int],
        invalidated_timestamps: set[bytes],
        const_transaction_fee: int,
        relative_transaction_fee: float,
        transaction_sane: bool = None,
        max_timestamp_now_diff: int = -1,
    ):
        can_afford = balances.get(self.sender, 0) > self.amount + self.fee > 0
        timestamp_valid = (
            max_timestamp_now_diff == -1
            or 0
            <= int.from_bytes(self.recv_time, ENDIAN)
            - int.from_bytes(self.timestamp, ENDIAN)
            <= max_timestamp_now_diff
        ) and self.timestamp not in invalidated_timestamps
        is_sane = (
            self.is_sane(const_transaction_fee, relative_transaction_fee)
            if transaction_sane is None
            else transaction_sane
        )
        conditions = (can_afford, timestamp_valid, is_sane)
        return all(conditions)

    @classmethod
    def from_bytes(cls, raw: bytes):
        components = []
        cum_size = 0
        for size, cf in zip(Transaction.BYTE_COUNTS, Transaction.CONVERSION_F):
            components.append(cf(raw[cum_size : cum_size + size]))
            cum_size += size
        return cls(*components)


class Block:
    SIZE = 4
    BYTE_COUNTS = [NONCE_SIZE_BYTES, 32, (PUBLIC_KEY_SIZE + 7) // 8, 4, 8] + [
        Transaction.N_BYTES
    ] * SIZE
    N_BYTES = sum(BYTE_COUNTS)
    CONVERSION_F = [
        lambda b: int.from_bytes(b, ENDIAN),
        LXX,
        LXX,
        lambda b: int.from_bytes(b, ENDIAN),
        lambda b: int.from_bytes(b, ENDIAN),
    ] + [Transaction.from_bytes] * SIZE

    def __init__(
        self,
        transactions: List["Transaction"],
        prev: Optional["Block"],
        validator: bytes,
        nonce: int = 0,
        version: int = 0,
        val_reward: int = 0,
    ):
        self.validator = validator
        self.transactions = transactions
        self.prev: Union[Optional["Block"], bytes] = prev
        self.nonce = nonce
        self.version = version
        self.val_reward = val_reward
        self._cache_hash = None
        self._cache_inner_hash = None
        self._cache_val = nonce

    def _bytes_without_nonce(self):
        return (
            (
                self.prev.hash()
                if self.prev is not None
                else b"\0" * Block.BYTE_COUNTS[1]
            )
            + self.validator
            + self.version.to_bytes(Block.BYTE_COUNTS[3], ENDIAN)
            + self.val_reward.to_bytes(Block.BYTE_COUNTS[4], ENDIAN)
            + b"".join(map(Transaction.to_bytes, self.transactions))
        )

    def to_bytes(self):
        return (
            self.nonce.to_bytes(Block.BYTE_COUNTS[0], ENDIAN)
            + self._bytes_without_nonce()
        )

    @classmethod
    def from_bytes(cls, raw: bytes):
        # NOTE: method assigns prev hash to prev member that should be Block object (gets linked externally)
        components = []
        cum_size = 0
        for size, cf in zip(Block.BYTE_COUNTS, Block.CONVERSION_F):
            components.append(cf(raw[cum_size : cum_size + size]))
            cum_size += size

        # last Block.SIZE components are transactions
        components, transactions = components[: -Block.SIZE], components[-Block.SIZE :]
        nonce, prev, validator, version, val_reward = components
        return cls(transactions, prev, validator, nonce, version, val_reward)

    def _inner_hash(self):
        if self._cache_inner_hash is None or True:
            self._cache_inner_hash = sha256(self._bytes_without_nonce())
        return self._cache_inner_hash

    def hash(self):
        if True or (self._cache_hash is None or self.nonce != self._cache_val):
            self._cache_hash = sha256(
                self.nonce.to_bytes(Block.BYTE_COUNTS[0], ENDIAN) + self._inner_hash()
            )
            self._cache_val = self.nonce
        return self._cache_hash


class NonceMiningJobHandler:
    def __init__(self):
        self.queued_jobs = []
        self._killed = threading.Event()
        self._job_killed = threading.Event()
        self.worker = threading.Thread(target=self._worker, daemon=True)
        self.worker.start()

    def launch_job(
        self,
        block: "Block",
        use_cuda_miner: bool,
        valid_block_max_hash: int,
        val_reward: int,
        init_nonce: int = 1,
    ):
        _global_lock.acquire()
        if any(b == block for b, *_ in self.queued_jobs):
            _global_lock.release()
            self.kill()
            raise ValueError("a job has already been queued for this block")
        self.queued_jobs.append(
            (block, use_cuda_miner, valid_block_max_hash, init_nonce, val_reward)
        )
        _global_lock.release()

    def kill(self):
        self._killed.set()
        self.worker.join()

    def kill_current_job(self):
        self._job_killed.set()

    def __del__(self):
        self.kill()

    @property
    def is_mining(self):
        return len(self.queued_jobs) > 0

    def _worker(self):
        while not self._killed.is_set():
            while not self.queued_jobs:
                if self._killed.is_set():
                    return
                if self._job_killed.is_set():
                    self._job_killed.clear()
                thread_yield()
            block, use_cuda_miner, valid_block_max_hash, init_nonce, val_reward = (
                self.queued_jobs[0]
            )
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
                    LinkedEvent(self._killed, self._job_killed),
                    init_nonce,
                )
                if nonce_out is not None:
                    block.nonce = nonce_out
                    if not is_valid_block(block, valid_block_max_hash):
                        raise RuntimeError(
                            f"Miner produced invalid nonce {hex(nonce_out)}. Please report to developer."
                        )
            else:
                block_ = Block(
                    block.transactions,
                    block.prev,
                    block.validator,
                    init_nonce,
                    block.version,
                    val_reward,
                )
                while (
                    not self._killed.is_set()
                    and not self._job_killed.is_set()
                    and not is_valid_block(block_, valid_block_max_hash)
                ):
                    block_.nonce = (block_.nonce + 1) % (1 << (NONCE_SIZE_BYTES * 8))
                if not self._killed.is_set() and not self._job_killed.is_set():
                    block.nonce = block_.nonce

            _global_lock.acquire()
            self.queued_jobs.pop(0)
            self._job_killed.clear()
            _global_lock.release()


def chain_from_bytes(raw: bytes):
    if raw == b"":
        return None
    block_bytes = [
        raw[i : i + Block.N_BYTES]
        for i in range(0, int(len(raw) / Block.N_BYTES) * Block.N_BYTES, Block.N_BYTES)
    ]
    if not all(len(bb) == Block.N_BYTES for bb in block_bytes):
        return None
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
            return None  # missing link in chain
        next_block = blocks[prevs.index(h)]
        next_block.prev = block
        blocks.remove(next_block)
        block = next_block
    return block


def chain_to_bytes(chain: "Block") -> bytes:
    return b"".join(block.to_bytes() for block in reversed_block_list(chain))


class TransactionPool:
    def __init__(
        self,
        chain: "Shared[Block]",
        validator: bytes,
        pool: Set["Transaction"] = None,
        get_padding_transaction: Optional[Callable[[], "Transaction"]] = None,
    ):
        if pool is None:
            pool = set()
        self.validator = validator
        self.chain = chain
        self.pool = pool
        self.nonce_mining_job_handler = NonceMiningJobHandler()
        self.get_padding_transaction_f = get_padding_transaction
        self._current_job = None

    def update_job(
        self,
        use_cuda_miner: bool,
        valid_block_max_hash: int,
        val_reward: int,
        init_nonce: int,
        balances: Dict[bytes, int],
        invalidated_timestamps: Set[bytes],
        const_transaction_fee: int,
        relative_transaction_fee: float,
        max_timestamp_now_diff: int,
        min_income_to_mine: int,
        version: int = 0,
    ):
        prev_block_was_ready_at_start = True
        if not block_is_ready(self.chain.x):
            prev_block_was_ready_at_start = False
            if balances.get(self.validator, 0) < val_reward:
                raise RuntimeError(
                    "Cannot undo coinbase-transaction: validator has too little currency. This is probably a bug. Please report to developer."
                )
            balances[self.validator] = balances.get(self.validator, 0) - val_reward
            for t in reversed(self.chain.x.transactions):
                undo_transaction(t, self.validator, balances, invalidated_timestamps)
                # they were removed from the pool when they were put into the last block, but since that is still
                # being mined and will be thrown away if there is a different set of transactions to be mined now,
                # I must add all these transactions back in.
                self.pool.add(t)

        while True:
            if self.get_padding_transaction_f is None and len(self.pool) < Block.SIZE:
                return
            transactions_to_mine = select_top_transactions(self.pool, Block.SIZE) + [
                self.get_padding_transaction_f()
                for _ in range(Block.SIZE - len(self.pool))
            ]

            for i, t in enumerate(transactions_to_mine):
                if not t.is_valid(
                    balances,
                    invalidated_timestamps,
                    const_transaction_fee,
                    relative_transaction_fee,
                    None,
                    max_timestamp_now_diff,
                ):
                    # remove invalid transaction and rerun
                    self.pool.remove(t)
                    for j in reversed(range(i)):
                        undo_transaction(
                            transactions_to_mine[j],
                            self.validator,
                            balances,
                            invalidated_timestamps,
                        )
                    break
                execute_transaction(t, self.validator, balances, invalidated_timestamps)

            else:
                if (
                    sum(
                        map(
                            lambda t: t.fee,
                            filter(lambda t: not t._is_padding, transactions_to_mine),
                        ),
                        start=val_reward,
                    )
                    >= min_income_to_mine
                ) and self._current_job != (self.chain.x, transactions_to_mine):
                    self._current_job = self.chain.x, transactions_to_mine
                    self.nonce_mining_job_handler.kill_current_job()
                    while self.nonce_mining_job_handler.is_mining:
                        thread_yield()

                    _global_lock.acquire()

                    prev_block_is_ready = block_is_ready(self.chain.x)
                    # avoid a race condition where the mining of prev block has finished during this computation
                    # since the transactions of that block were added back to the pool in preparation of the running
                    # block being killed and replaced by a better one (other transactions to be mined).
                    # however, now those transactions are already in the blockchain and must not be considered.
                    # we have to rerun without considering these transactions.
                    if prev_block_is_ready and not prev_block_was_ready_at_start:
                        # reset all changes made to the balances in this loop run
                        for t in reversed(transactions_to_mine):
                            undo_transaction(
                                t, self.validator, balances, invalidated_timestamps
                            )
                        for t in self.chain.x.transactions:
                            self.pool.remove(t)
                            execute_transaction(
                                t, self.validator, balances, invalidated_timestamps
                            )
                        _global_lock.release()
                        continue

                    # remove selected transactions from pool since they are now in new block
                    for t in transactions_to_mine:
                        self.pool.remove(t)

                    self.chain.x = Block(
                        transactions_to_mine,
                        (  # if last block is work-in-progress, throw it away and replace it, otherwise, extend chain
                            self.chain.x if prev_block_is_ready else self.chain.x.prev
                        ),
                        self.validator,
                        version=version,
                        val_reward=val_reward,
                    )
                    _global_lock.release()

                    self.nonce_mining_job_handler.launch_job(
                        self.chain.x,
                        use_cuda_miner,
                        valid_block_max_hash,
                        val_reward,
                        init_nonce,
                    )
                break

    def receive(
        self,
        transaction: "Transaction",
        balances: Dict[bytes, int],
        invalidated_timestamps: set[bytes],
        const_transaction_fee: int,
        relative_transaction_fee: float,
        use_cuda_miner: bool,
        valid_block_max_hash: int,
        val_reward: int,
        init_nonce: int,
        min_income_to_mine: int,
        transaction_sane: bool = None,
        max_timestamp_now_diff: int = -1,
        version: int = 0,
    ) -> tuple[bool, "FeedbackCode"]:
        transaction.recv_time = time.time_ns().to_bytes(
            Transaction.BYTE_COUNTS[5], ENDIAN
        )
        if (
            transaction_sane is None
            and not transaction.is_sane(const_transaction_fee, relative_transaction_fee)
            or transaction_sane is False
        ):
            return False, FeedbackCode.TRANSACTION_INSANE
        self.pool.add(transaction)
        self.update_job(
            use_cuda_miner,
            valid_block_max_hash,
            val_reward,
            init_nonce,
            balances,
            invalidated_timestamps,
            const_transaction_fee,
            relative_transaction_fee,
            max_timestamp_now_diff,
            min_income_to_mine,
            version,
        )
        accepted = transaction in self.pool
        return accepted, (
            FeedbackCode.TRANSACTION_ACCEPTED
            if accepted
            else FeedbackCode.TRANSACTION_REJECTED_BUT_SANE
        )


def get_received_transactions(
    transaction_buffer, n: int
) -> tuple[List["Transaction"], List[bytes]]:
    if transaction_buffer is None:
        return [], []
    out = []
    out_bytes = []
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
                out_bytes.append(raw)
                n -= 1
    else:
        while n and transaction_buffer:
            raw = transaction_buffer.pop(0)
            if len(raw) != Transaction.N_BYTES:
                continue
            out.append(Transaction.from_bytes(raw))
            out_bytes.append(raw)
            n -= 1
    return out, out_bytes


def get_received_chains(chain_buffer, n: int) -> tuple[List["Block"], List[bytes]]:
    if chain_buffer is None:
        return [], []
    out = []
    out_bytes = []
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
                    out_bytes.append(raw)
                    n -= 1
    else:
        while n and chain_buffer:
            raw = chain_buffer.pop(0)
            if len(raw) % Block.N_BYTES != 0:
                continue
            chain = chain_from_bytes(raw)
            if chain is not None:
                out.append(chain)
                out_bytes.append(raw)
                n -= 1
    return out, out_bytes


def get_received_blocks(block_buffer, n: int) -> tuple[List["Block"], List[bytes]]:
    if block_buffer is None:
        return [], []
    out = []
    out_bytes = []
    if isinstance(block_buffer, (queue.Queue, mp.queues.Queue)):
        while not block_buffer.empty() and n:
            try:
                raw = block_buffer.get_nowait()
            except queue.Empty:
                pass
            else:
                if len(raw) != Block.N_BYTES:
                    continue
                block = Block.from_bytes(raw)
                if block is not None and len(block.transactions) == Block.SIZE:
                    out.append(block)
                    out_bytes.append(raw)
                    n -= 1
    else:
        while n and block_buffer:
            raw = block_buffer.pop(0)
            if len(raw) != Block.N_BYTES:
                continue
            block = chain_from_bytes(raw)
            if block is not None and len(block.transactions) == Block.SIZE:
                out.append(block)
                out_bytes.append(raw)
                n -= 1
    return out, out_bytes


def broadcast_chain(broadcast_buffer, block: "Block", broadcast_chain_as_bytes: bool):
    if broadcast_buffer is None:
        return
    chain_bytes = chain_to_bytes(block) if broadcast_chain_as_bytes else block
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
    block_recv_channel=None,
    chain_broadcast_channel=None,
    balance_broadcast_channel=None,
    side_channel=None,
    init_balances=None,
    terminate_event=None,
    has_terminated_event=None,
    use_cuda_miner_event=None,
    incompatible_chain_distrust: int = 1,
    compatible_chain_distrust: int = 1,
    val_reward: int = 0,
    const_min_transaction_fee: int = 0,
    relative_min_transaction_fee: float = 0.0,
    max_recv_chains_per_iter: int = -1,
    max_recv_blocks_per_iter: int = -1,
    max_recv_transactions_per_iter: int = -1,
    collect_time: int = 0,
    max_timestamp_now_diff: int = -1,
    copy_balances_on_broadcast=False,
    transaction_backlog_size: int = 1,
    broadcast_balances_as_bytes: bool = False,
    broadcast_chain_as_bytes: bool = False,
    valid_block_max_hash: int = 0x00000000FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF,
    version: int = 0,
    cuda_miner_load_args: Optional[tuple] = None,
    get_padding_transaction: Optional[Callable[[], "Transaction"]] = None,
    min_income_to_mine: int = 1,
):
    """
    The main blockchain function. It receives transactions and chains, verifies them, and updates the blockchain.
    :param validator: the public key of the miner
    :param chain: optional initial chain
    :param transaction_recv_channel: queue for receiving transactions
    :param chain_recv_channel: queue for receiving chains
    :param block_recv_channel: queue for receiving blocks
    :param chain_broadcast_channel: queue for broadcasting new chains that a nonce was found for
    :param balance_broadcast_channel: queue for broadcasting balances for review
    :param side_channel: queue for broadcasting expensive to compute data that might be reused (eg chain validity)
    :param init_balances: initial balance of the blockchain (what the balance is before the first transaction)
    :param terminate_event: event set when the blockchain main loop should terminate
    :param has_terminated_event: event set when the blockchain main loop has terminated
    :param use_cuda_miner_event: event that signals that the miner should use the CUDA miner kernel instead of the debug miner
    :param incompatible_chain_distrust: how many blocks a chain that has incompatible balances with the current chain must be longer to replace the current chain
    :param compatible_chain_distrust: how many blocks a chain that has the same balances at the end has to be longer to replace the current chain
    :param val_reward: the reward for validating a block
    :param const_min_transaction_fee: the constant transaction fee that is subtracted from the amount sent and given to the validator
    :param relative_min_transaction_fee: percentage (value between 0 and 1) of the amount sent that is given to the validator
    :param max_recv_chains_per_iter: how many alternative chains may be processed in one loop iteration (helps limit time / iter). negative value means no limit
    :param max_recv_blocks_per_iter: how many blocks may be processed in one loop iteration (helps limit time / iter). negative value means no limit
    :param max_recv_transactions_per_iter: how many transactions may be processed in one loop iteration (helps limit time / iter). negative value means no limit
    :param collect_time: how many nanoseconds to wait and collect transactions before processing them (allows waiting for more transactions to arrive so miner can select the most valuable ones)
    :param max_timestamp_now_diff: the max amount of time a transaction timestamp can be behind the current time
    :param copy_balances_on_broadcast: whether to copy the balances dictionary when broadcasting it. (helps prevent bugs by removing mutation coupling with blockchain core). Ignored if broadcast_balances_as_bytes is True.
    :param transaction_backlog_size: how many blocks' submitted transactions are stored in the backlog in case an alternative chain arrives (where those suddenly are valid)
    :param broadcast_balances_as_bytes: whether to broadcast balances as bytes or as a dictionary
    :param broadcast_chain_as_bytes: whether to broadcast chains as bytes or as a Block object
    :param valid_block_max_hash: the max value of the hash (interpreted as integer) that makes a block valid
    :param version: the version of the blockchain (to allow for forks)
    :param cuda_miner_load_args: Optional tuple or args used to (optionally) load a cuda miner on startup
    :param get_padding_transaction: Optional function that takes no args and returns a new padding transaction (= transaction from validator_wallet_1 to validator_wallet_2)
    :param min_income_to_mine: The minimum amount of income a miner has to make from the transaction pool to mine it (to avoid power consumption exceeding gain)
    """
    # TODO add parameter for TransactionPool, send out TransactionPool via side channel when terminating
    # and save TransactionPool and pass it back in when running this function again. Also verify validity of all
    # transactions in the pool.

    if cuda_miner_load_args is not None:
        load_cuda_miner(*cuda_miner_load_args)

    assert isinstance(version, int), "version must be integer"
    assert transaction_backlog_size >= 0, "transaction_backlog_size cannot be negative"
    assert max_recv_chains_per_iter != 0, "max_recv_chains_per_iter cannot be 0"
    assert max_recv_blocks_per_iter != 0, "max_recv_blocks_per_iter cannot be 0"
    assert isinstance(min_income_to_mine, int)
    assert min_income_to_mine >= 0
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
    assert const_min_transaction_fee >= 0, "you cannot have a negative transaction fee"
    assert (
        0 <= relative_min_transaction_fee < 1
    ), "relative transaction fee must be in [0, 1)"
    assert len(validator) == PUBLIC_KEY_SIZE // 8, "invalid validator public key size"
    assert transaction_recv_channel is None or isinstance(
        transaction_recv_channel, (queue.Queue, mp.queues.Queue, list)
    )
    assert chain_recv_channel is None or isinstance(
        chain_recv_channel, (queue.Queue, mp.queues.Queue, list)
    )
    assert block_recv_channel is None or isinstance(
        block_recv_channel, (queue.Queue, mp.queues.Queue, list)
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
    assert init_balances is None or isinstance(init_balances, dict)
    assert terminate_event is None or isinstance(
        terminate_event, (threading.Event, mp.synchronize.Event)
    )
    assert has_terminated_event is None or isinstance(
        has_terminated_event, (threading.Event, mp.synchronize.Event)
    )
    assert use_cuda_miner_event is None or isinstance(
        use_cuda_miner_event, (threading.Event, mp.synchronize.Event)
    )
    assert get_padding_transaction is None or callable(get_padding_transaction)
    assert isinstance(incompatible_chain_distrust, int)
    assert isinstance(compatible_chain_distrust, int)
    assert isinstance(val_reward, int)
    assert isinstance(const_min_transaction_fee, int)
    assert isinstance(relative_min_transaction_fee, float)
    assert isinstance(max_recv_chains_per_iter, int)
    assert isinstance(max_recv_transactions_per_iter, int)
    assert isinstance(collect_time, int)
    assert isinstance(copy_balances_on_broadcast, bool)
    assert isinstance(transaction_backlog_size, int)
    assert isinstance(broadcast_balances_as_bytes, bool)
    assert isinstance(broadcast_chain_as_bytes, bool)
    assert isinstance(validator, bytes)
    assert isinstance(chain, Block) or chain is None
    assert isinstance(init_balances, dict) or init_balances is None

    vsc = verified_subchain(chain)
    init_chain_valid, balances, invalidated_timestamps = full_verify(
        vsc,
        init_balances,
        val_reward,
        const_min_transaction_fee,
        relative_min_transaction_fee,
        valid_block_max_hash,
        max_timestamp_now_diff,
    )
    if not init_chain_valid:
        raise RuntimeError("invalid initial chain provided")

    if RUN_WEAKNESS_DETECTION_WARNINGS:
        if val_reward != 0:
            print(HAS_VAL_REWARD_WARNING)

    if init_balances is None:
        init_balances = {}

    chain = Shared(chain)
    transaction_pool = TransactionPool(chain, validator)
    transaction_backlog = [[]] * transaction_backlog_size
    updated_chain = False
    replaced_chain = False
    last_loop_time_ns = 0
    last_block_balances = balances.copy()
    block_hashes = [block.hash() for block in block_list(chain.x)]

    if chain.x != vsc:
        raise ValueError("initial chain has unverified blocks")

    while terminate_event is None or not terminate_event.is_set():
        while time.time_ns() - last_loop_time_ns < collect_time and (
            terminate_event is None or not terminate_event.is_set()
        ):
            thread_yield()
        last_loop_time_ns = time.time_ns()
        current_chain_len = chain_len(chain.x)

        # receive new chains
        recv_chains, recv_chain_bytes = get_received_chains(
            chain_recv_channel, max_recv_chains_per_iter
        )
        favorite_new_chain_data = (None, None, None)
        max_trust_level = 0
        for recv_chain, chain_bytes in zip(recv_chains, recv_chain_bytes):
            recv_chain = verified_subchain(recv_chain)
            new_chain_len = chain_len(recv_chain)
            is_valid, new_balances, new_invalidated_timestamps = full_verify(
                recv_chain,
                init_balances,
                val_reward,
                const_min_transaction_fee,
                relative_min_transaction_fee,
                valid_block_max_hash,
                max_timestamp_now_diff,
            )
            if side_channel is not None and not is_valid:
                broadcast_side_channel(
                    side_channel,
                    SideChannelItemType.CHAIN_FEEDBACK_PAIR,
                    (chain_bytes, FeedbackCode.CHAIN_INVALID),
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
                broadcast_side_channel(
                    side_channel,
                    SideChannelItemType.CHAIN_FEEDBACK_PAIR,
                    (chain_bytes, FeedbackCode.CHAIN_ACCEPTED),
                )
                favorite_new_chain_data = (
                    recv_chain,
                    new_balances,
                    new_invalidated_timestamps,
                )
                max_trust_level = max(
                    new_chain_len - current_chain_len - incompatible_chain_distrust + 1,
                    new_chain_len - current_chain_len - compatible_chain_distrust + 1,
                )
            else:
                broadcast_side_channel(
                    side_channel,
                    SideChannelItemType.CHAIN_FEEDBACK_PAIR,
                    (chain_bytes, FeedbackCode.CHAIN_VALID_BUT_NOT_ACCEPTED),
                )
        if favorite_new_chain_data[0] is not None:
            chain, balances, invalidated_timestamps = favorite_new_chain_data
            chain = Shared(chain)
            last_block_balances = balances.copy()
            transaction_pool = TransactionPool(chain, validator)
            replaced_chain = True
            updated_chain = True

        # receive new blocks (to reduce network traffic)
        recv_blocks, recv_block_bytes = get_received_blocks(
            block_recv_channel, max_recv_blocks_per_iter
        )
        for recv_block, block_bytes in zip(recv_blocks, recv_block_bytes):
            # block hasn't been linked yet, so prev is just the hash not the block obj
            prev_block_hash = recv_block.prev
            if prev_block_hash != (
                b"\0" * Block.BYTE_COUNTS[1] if chain is None else chain.hash()
            ):
                if side_channel is not None:
                    broadcast_side_channel(
                        side_channel,
                        SideChannelItemType.BLOCK_FEEDBACK_PAIR,
                        (block_bytes, FeedbackCode.BLOCK_REJECTED),
                    )
                continue

            fake_block = Block(*((None,) * 6))
            fake_block.hash = lambda: prev_block_hash
            recv_block.prev = fake_block
            is_valid, new_balances, new_invalidated_timestamps = full_verify(
                recv_block,
                last_block_balances,
                val_reward,
                const_min_transaction_fee,
                relative_min_transaction_fee,
                valid_block_max_hash,
                max_timestamp_now_diff,
                _depth=1,
            )

            if side_channel is not None:
                broadcast_side_channel(
                    side_channel,
                    SideChannelItemType.BLOCK_FEEDBACK_PAIR,
                    (
                        block_bytes,
                        (
                            FeedbackCode.BLOCK_ACCEPTED
                            if is_valid
                            else FeedbackCode.BLOCK_REJECTED
                        ),
                    ),
                )
            if is_valid:
                recv_block.prev = chain.x
                chain = Shared(recv_block)
                balances = new_balances
                last_block_balances = balances.copy()
                invalidated_timestamps = new_invalidated_timestamps
                transaction_pool = TransactionPool(chain, validator)
                replaced_chain = True
                updated_chain = True

        # receive new transactions
        recv_trans, recv_trans_bytes = get_received_transactions(
            transaction_recv_channel, max_recv_transactions_per_iter
        )
        t_is_sane = [
            t.is_sane(const_min_transaction_fee, relative_min_transaction_fee)
            for t in recv_trans
        ]
        if side_channel is not None:
            for t_bytes, sane in zip(recv_trans_bytes, t_is_sane):
                if not sane:
                    broadcast_side_channel(
                        side_channel,
                        SideChannelItemType.TRANSACTION_FEEDBACK_PAIR,
                        (t_bytes, FeedbackCode.TRANSACTION_INSANE),
                    )
        filtered_recv_transactions_and_bytes = [
            (t, t_bytes)
            for t, t_bytes, sane in zip(recv_trans, recv_trans_bytes, t_is_sane)
            if sane
        ]
        transactions = sorted(
            filtered_recv_transactions_and_bytes,
            key=lambda t_and_bytes: int.from_bytes(t_and_bytes[0].timestamp, ENDIAN),
        )
        if replaced_chain:
            transactions = sum(transaction_backlog, start=[]) + transactions
        for transaction, t_bytes in transactions:
            if transaction_backlog_size:
                transaction_backlog[-1].append(transaction)
            is_valid_t = transaction_pool.receive(
                transaction,
                balances,
                invalidated_timestamps,
                const_min_transaction_fee,
                relative_min_transaction_fee,
                use_cuda_miner_event is not None and use_cuda_miner_event.is_set(),
                valid_block_max_hash,
                val_reward,
                int.from_bytes(random.randbytes(NONCE_SIZE_BYTES), ENDIAN),
                min_income_to_mine,
                True,
                max_timestamp_now_diff,
                version,
            )
            if side_channel is not None:
                broadcast_side_channel(
                    side_channel,
                    SideChannelItemType.TRANSACTION_FEEDBACK_PAIR,
                    (
                        t_bytes,
                        (
                            FeedbackCode.TRANSACTION_ACCEPTED
                            if is_valid_t
                            else FeedbackCode.TRANSACTION_REJECTED_BUT_SANE
                        ),
                    ),
                )
            last_block_balances = balances.copy()
            # remove those that were just closed
            transaction_backlog.pop(0)
            transaction_backlog.append([])
            updated_chain = updated_chain or current_chain_len != chain_len(chain.x)

        new_block_hashes = [block.hash() for block in block_list(chain.x)]
        if new_block_hashes != block_hashes:
            block_hashes = new_block_hashes
            broadcast_chain(chain_broadcast_channel, chain.x, broadcast_chain_as_bytes)
            broadcast_balances(
                balance_broadcast_channel,
                last_block_balances,
                copy_balances_on_broadcast,
                broadcast_balances_as_bytes,
            )
            updated_chain = False

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
        max_recv_blocks_per_iter: int = 1,
        max_recv_transactions_per_iter: int = Block.SIZE,
        collect_time: int = 0,
        max_timestamp_now_diff: int = -1,
        copy_balances_on_broadcast: bool = True,
        transaction_backlog_size: int = 2,
        broadcast_balances_as_bytes: bool = True,
        broadcast_chain_as_bytes: bool = True,
        valid_block_max_hash: int = 0x00000000FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF,
        version: int = 0,
        min_income_to_mine: int = 1,
    ):
        self.incompatible_chain_distrust = incompatible_chain_distrust
        self.compatible_chain_distrust = compatible_chain_distrust
        self.val_reward = val_reward
        self.const_transaction_fee = const_transaction_fee
        self.relative_transaction_fee = relative_transaction_fee
        self.max_recv_chains_per_iter = max_recv_chains_per_iter
        self.max_recv_blocks_per_iter = max_recv_blocks_per_iter
        self.max_recv_transactions_per_iter = max_recv_transactions_per_iter
        self.collect_time = collect_time
        self.max_timestamp_now_diff = max_timestamp_now_diff
        self.copy_balances_on_broadcast = copy_balances_on_broadcast
        self.transaction_backlog_size = transaction_backlog_size
        self.broadcast_balances_as_bytes = broadcast_balances_as_bytes
        self.broadcast_chain_as_bytes = broadcast_chain_as_bytes
        self.valid_block_max_hash = valid_block_max_hash
        self.version = version
        self.min_income_to_mine = min_income_to_mine


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
        init_balances=None,
        config: MiningConfig = None,
        run_with: str = "direct",
        start_immediately: bool = True,
        get_padding_transaction: Optional[Callable[[], "Transaction"]] = None,
    ):
        super().__init__()
        self.validator = validator
        self.chain = chain
        self.init_balances = init_balances.copy() if init_balances is not None else {}
        self.get_padding_transaction = get_padding_transaction
        if config is None:
            config = MiningConfig()

        if run_with == "threading":
            self.transaction_recv_buffer = queue.Queue()
            self.chain_recv_buffer = queue.Queue()
            self.block_recv_buffer = queue.Queue()
            self.chain_broadcast_buffer = queue.Queue(1)
            self.balance_broadcast_buffer = queue.Queue(1)
            self.side_channel = queue.Queue()
            self.terminate_event = threading.Event()
            self.has_terminated_event = threading.Event()
            self.use_cuda_miner_event = threading.Event()
        elif run_with == "mp":
            self.transaction_recv_buffer = mp.Queue()
            self.chain_recv_buffer = mp.Queue()
            self.block_recv_buffer = mp.Queue()
            self.chain_broadcast_buffer = mp.Queue(1)
            self.balance_broadcast_buffer = mp.Queue(1)
            self.side_channel = mp.Queue()
            self.terminate_event = mp.Event()
            self.has_terminated_event = mp.Event()
            self.use_cuda_miner_event = mp.Event()
        else:
            self.transaction_recv_buffer = queue.Queue()
            self.chain_recv_buffer = queue.Queue()
            self.block_recv_buffer = queue.Queue()
            self.chain_broadcast_buffer = queue.Queue(1)
            self.balance_broadcast_buffer = queue.Queue(1)
            self.side_channel = queue.Queue()
            self.terminate_event = threading.Event()
            self.use_cuda_miner_event = threading.Event()

        self.incompatible_chain_distrust = config.incompatible_chain_distrust
        self.compatible_chain_distrust = config.compatible_chain_distrust
        self.val_reward = config.val_reward
        self.const_transaction_fee = config.const_transaction_fee
        self.relative_transaction_fee = config.relative_transaction_fee
        self.max_recv_chains_per_iter = config.max_recv_chains_per_iter
        self.max_recv_blocks_per_iter = config.max_recv_blocks_per_iter
        self.max_recv_transactions_per_iter = config.max_recv_transactions_per_iter
        self.collect_time = config.collect_time
        self.max_timestamp_now_diff = config.max_timestamp_now_diff
        self.copy_balances_on_broadcast = config.copy_balances_on_broadcast
        self.transaction_backlog_size = config.transaction_backlog_size
        self.broadcast_balances_as_bytes = config.broadcast_balances_as_bytes
        self.broadcast_chain_as_bytes = config.broadcast_chain_as_bytes
        self.valid_block_max_hash = config.valid_block_max_hash
        self.version = config.version
        self.min_income_to_mine = config.min_income_to_mine

        self.run_with = run_with
        self._get_balances_last = (
            serialize_balances(self.init_balances)
            if self.broadcast_balances_as_bytes
            else (
                self.init_balances.copy()
                if self.copy_balances_on_broadcast
                else self.init_balances
            )
        )
        self._get_chain_last = (
            chain_to_bytes(self.chain)
            if self.broadcast_chain_as_bytes and self.chain is not None
            else b"" if self.broadcast_chain_as_bytes else self.chain
        )
        self.args = (
            self.validator,
            self.chain,
            self.transaction_recv_buffer,
            self.chain_recv_buffer,
            self.block_recv_buffer,
            self.chain_broadcast_buffer,
            self.balance_broadcast_buffer,
            self.side_channel,
            self.init_balances,
            self.terminate_event,
            self.has_terminated_event,
            self.use_cuda_miner_event,
            self.incompatible_chain_distrust,
            self.compatible_chain_distrust,
            self.val_reward,
            self.const_transaction_fee,
            self.relative_transaction_fee,
            self.max_recv_chains_per_iter,
            self.max_recv_blocks_per_iter,
            self.max_recv_transactions_per_iter,
            self.collect_time,
            self.max_timestamp_now_diff,
            self.copy_balances_on_broadcast,
            self.transaction_backlog_size,
            self.broadcast_balances_as_bytes,
            self.broadcast_chain_as_bytes,
            self.valid_block_max_hash,
            self.version,
            (
                _cuda_miner_load_callback_args
                if run_with == "mp" or _cuda_miner_mod is None
                else None
            ),
            self.get_padding_transaction,
            self.min_income_to_mine,
        )
        self.started = start_immediately
        if run_with == "direct":
            if start_immediately:
                _mine(*self.args)
        elif run_with == "threading":
            self._thread = threading.Thread(target=_mine, daemon=True, args=self.args)
            if start_immediately:
                self._thread.start()
        elif run_with == "mp":
            self._process = mp.Process(target=_mine, daemon=True, args=self.args)
            if start_immediately:
                self._process.start()
        else:
            raise ValueError(
                "invalid run_with value: run_with must be one of ('direct', 'threading', 'mp')"
            )

    def start(self):
        if self.started:
            raise RuntimeError("Miner already started")
        elif self.run_with == "direct":
            _mine(*self.args)
        elif self.run_with == "threading":
            self._thread.start()
        else:
            self._process.start()

    def to_bytes(self):
        return self.validator + chain_to_bytes(self.chain)

    @classmethod
    def from_bytes(cls, raw: bytes, **kwargs):
        validator, chain_bytes = (
            raw[: PUBLIC_KEY_SIZE // 8],
            raw[PUBLIC_KEY_SIZE // 8 :],
        )
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

    def add_transaction(self, transaction: bytes):
        self.transaction_recv_buffer.put_nowait(transaction)

    def add_chain(self, chain: bytes):
        self.chain_recv_buffer.put_nowait(chain)

    def add_block(self, block: bytes):
        self.block_recv_buffer.put_nowait(block)

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
