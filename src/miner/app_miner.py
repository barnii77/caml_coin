import os
import string
import functools
import json
import time
import hashlib
import secrets
import signal
import requests
import multiprocessing as mp
import threading
import argparse
from typing import Optional
from flask import Flask, request, render_template, redirect, url_for

# TODO benchmark blockchain engine because it is terribly slow and unit tests don't always pass

# TODO change the known miner removal system (for inactive miners) so that all "did he respond" bools over a certain
# amount of time are saved, regularly filtered (like the caches) and the cutoff for removal is a percentage of the samples
# and a minimum amount of samples required for significance + a maximum amount of amount of samples where the miner is
# considered bombarded with messages and we are forgiving with him for not answering.

# TODO fix/test invalid block full chain request callback

# TODO encrypt private key when it is generated and on startup, require entering a password to decrypt it; also check it is correct by deriving the public key from decrypted private key and checking equivalence (public is stored plainly)

# TODO figure out how to dynamically adjust the valid_block_max_hash (difficulty)

_chain_time_cache = {}
_block_time_cache = {}
_transaction_time_cache = {}
_miner_n_non_responsive = {}
_hash_to_validity = {}
_hash_to_save_time = {}
_temporary_blacklist = {}
_hash_to_sender = {}

REQUEST_TIMEOUT = 10
KEEP_IN_CACHE_SECS = 60
CACHE_CLEAN_DELAY_SECS = 5
REGISTER_VALIDITY_DELAY_SECS = 3
BROADCAST_THREAD_POOL_SIZE = 10
OTHER_MINER_CONSIDER_INVALID_NUM_UNRESPONSIVE = 5
VALIDITY_RETAIN_TIME = 300
TEMPORARY_BLACKLIST_TIME = 30
CHAIN_SAVE_DELAY_SECS = 1440 * 60
MINER_SAVE_DELAY_SECS = 1440 * 60
UUID_SIZE_BYTES = 8
LN1 = lambda _: None

with open("known_miners.txt", "r") as f:
    other_miners = set(line.rstrip() for line in f.readlines())


def schedule(task, delay, num_repeats=-1):
    """
    Runs task num_repeats times (or indefinitely if num_repeats < 0), waiting for delay seconds before every execution.
    """

    def task_wrapper():
        n = num_repeats
        while n:
            time.sleep(delay)
            task()
            if n > 0:
                n -= 1

    threading.Thread(target=task_wrapper, daemon=True).start()


def save_chain():
    if not save_data_switch.is_set():
        return
    chain = miner.get_chain()
    if chain is not None:
        with open("chains/chain.bin", "wb") as f:
            f.write(miner.validator + chain)


def save_known_miners():
    if not save_data_switch.is_set():
        return
    with open("known_miners.txt", "w") as f:
        f.write("\n".join(other_miners))


def add_miner_response_sample(sender, sample):
    _miner_n_non_responsive[sender].append(sample)


def _broadcast_to_miner(m: str, data: bytes, route: str):
    print("broadcasting to", m)
    try:
        requests.post(f"http://{m}/{route}", data=data, timeout=REQUEST_TIMEOUT)
    except requests.exceptions.RequestException:
        # TODO use add_miner_response_sample
        _miner_n_non_responsive[m] = _miner_n_non_responsive.get(m, 0) + 1


def split_set_into_n_parts(set_: set, n_parts: int):
    if not set_:
        return []
    n = len(set_) // n_parts + int(len(set_) % n_parts != 0)
    result = []
    iterator = iter(set_)
    for i in range(0, len(set_), n):
        result.append([next(iterator) for _ in range(min(n, len(set_) - i))])
    return result


def broadcast_to_miners(data: bytes, route: str):
    """Broadcasts data to all miners in the network (it knows)."""
    # for m in other_miners:
    #     _broadcast_to_miner(m, data, route)
    broadcaster_thread_pool = [
        threading.Thread(
            target=lambda miners: [_broadcast_to_miner(m, data, route) for m in miners],
            args=(miner_split,),
            daemon=True,
        )
        for miner_split in split_set_into_n_parts(
            other_miners, BROADCAST_THREAD_POOL_SIZE
        )
    ]
    for t in broadcaster_thread_pool:
        t.start()
    for t in broadcaster_thread_pool:
        t.join()


def clean_caches():
    now = time.time()
    remove_keys = []
    caches = [
        _chain_time_cache,
        _block_time_cache,
        _transaction_time_cache,
        _miner_n_non_responsive,
        _hash_to_save_time,
        _temporary_blacklist,
    ]
    remove_callbacks = [
        LN1,
        LN1,
        LN1,
        lambda other_miner: other_miners.remove(other_miner),
        lambda h: _hash_to_validity.pop(h),
        LN1,
    ]
    dropout_time = [
        KEEP_IN_CACHE_SECS,
        KEEP_IN_CACHE_SECS,
        KEEP_IN_CACHE_SECS,
        OTHER_MINER_CONSIDER_INVALID_NUM_UNRESPONSIVE,
        VALIDITY_RETAIN_TIME,
        TEMPORARY_BLACKLIST_TIME,
    ]
    for cache, retain_time, remove_callback in zip(
        caches, dropout_time, remove_callbacks
    ):
        for x, t in cache.items():
            if now - t > retain_time:
                remove_keys.append(x)
        for x in remove_keys:
            remove_callback(x)
            cache.pop(x)
        remove_keys.clear()


def register_new_validities():
    def blacklist(sender):
        _temporary_blacklist[sender] = time.time()

    def request_full_chain(sender):
        print("trying to get full chain")
        try:
            chain = requests.get(f"http://{sender}/get-chain", timeout=REQUEST_TIMEOUT).content
        except requests.exceptions.RequestException:
            if sender in other_miners:
                # TODO use add_miner_response_sample
                _miner_n_non_responsive[sender] = _miner_n_non_responsive.get(sender, 0)
            else:
                blacklist(sender)
            return
        _hash_to_sender[blockchain.sha256(chain)] = sender
        print("redirecting full chain to localhost")
        try:
            requests.post(
                f"http://localhost:{args.port}/submit-chain",
                data=chain,
                timeout=REQUEST_TIMEOUT,
            )
        except requests.exceptions.RequestException:
            pass

    item_type_to_func_if_invalid = [
        blacklist,
        blacklist,
        request_full_chain,
    ]
    item = miner.get_side_channel_item()
    while item is not None:
        item_type, (data, validity) = item
        print("executing a callback of type", item_type)
        h = blockchain.sha256(data)
        _hash_to_validity[h] = validity
        _hash_to_save_time[h] = time.time()
        if not validity:
            sender = _hash_to_sender[h]
            item_type_to_func_if_invalid[item_type.value](sender)
        else:
            pass  # TODO broadcast to other miners here
        item = miner.get_side_channel_item()


# TODO these functions currently broadcast to miners, but let the callbacks do that once I am sure something is valid
# this is so I avoid being blacklisted by everyone
def add_chain(chain: bytes):
    _hash_to_sender.setdefault(blockchain.sha256(chain), request.remote_addr)
    if chain not in _chain_time_cache:
        miner.add_chain(chain)
        _chain_time_cache[chain] = time.time()
        broadcast_to_miners(chain, "submit-chain")


def add_block(block: bytes):
    _hash_to_sender.setdefault(blockchain.sha256(block), request.remote_addr)
    if block not in _block_time_cache:
        miner.add_block(block)
        _block_time_cache[block] = time.time()
        broadcast_to_miners(block, "submit-block")


def add_transaction(transaction: bytes):
    _hash_to_sender.setdefault(blockchain.sha256(transaction), request.remote_addr)
    if transaction not in _transaction_time_cache:
        miner.add_transaction(transaction)
        _transaction_time_cache[transaction] = time.time()
        broadcast_to_miners(transaction, "submit-transaction")


def disallow_blacklisted(func):
    @functools.wraps(func)
    def wrapper():
        if request.remote_addr not in _temporary_blacklist:
            return func()
        return redirect(url_for("route_blacklisted")), 403

    return wrapper


app = Flask(__name__)


@app.route("/")
@disallow_blacklisted
def index():  # put application's code here
    return (
        render_template(
            "index.html",
            validator=miner.validator.hex(),
            len=len,
            VALIDATOR_CHARS_PER_LINE=100,
        ),
        200,
    )


@app.route("/shutdown", methods=["GET", "POST"])
@disallow_blacklisted
def shutdown():
    if request.method == "GET":
        return render_template("shutdown.html"), 200
    else:
        password = request.form.get("password", request.data.decode("utf-8"))
        pw_hash = hashlib.sha256((password + shutdown_pw_salt).encode()).hexdigest()
        if pw_hash != shutdown_pw_hash:
            return redirect("/shutdown"), 403
        save_data_switch.clear()  # avoid race condition with automatic chain saving
        chain, _ = miner.finish()
        if chain is not None:
            with open("chains/chain.bin", "wb") as f:
                f.write(miner.validator + chain)

        schedule(lambda: os.kill(app_pid, signal.SIGTERM), 5, 1)
        return "<h1>Server shutting down</h1>", 200


@app.route("/submit-transaction-hex", methods=["GET", "POST"])
@disallow_blacklisted
def submit_transaction_hex():
    if request.method == "GET":
        # render an html template that lets the user submit a transaction in hex
        return render_template("submit_transaction.html"), 200
    else:
        transaction = (
            bytes.fromhex(request.form["transaction"])
            if "transaction" in request.form
            else bytes.fromhex(str(request.data))
        )
        add_transaction(transaction)
        return redirect(url_for("index")), 200


@app.route("/submit-chain-hex", methods=["GET", "POST"])
@disallow_blacklisted
def submit_chain_hex():
    if request.method == "GET":
        return render_template("submit_chain.html"), 200
    else:
        chain = (
            bytes.fromhex(request.form["chain"])
            if "chain" in request.form
            else bytes.fromhex(str(request.data))
        )
        add_chain(chain)
        return redirect(url_for("index")), 200


@app.route("/submit-block-hex", methods=["GET", "POST"])
@disallow_blacklisted
def submit_block_hex():
    if request.method == "GET":
        return render_template("submit_block.html"), 200
    else:
        block = (
            bytes.fromhex(request.form["block"])
            if "block" in request.form
            else bytes.fromhex(str(request.data))
        )
        add_block(block)
        return redirect(url_for("index")), 200


@app.route("/get-chain-hex", methods=["GET"])
@disallow_blacklisted
def get_chain_hex():
    chain: Optional[bytes] = miner.get_chain()
    if chain is None:
        return "No chain found", 404
    return chain.hex(), 200


@app.route("/get-balances-hex", methods=["GET"])
@disallow_blacklisted
def get_balances_hex():
    balances_raw: Optional[dict[bytes, int]] = miner.get_balances()
    if balances_raw is None:
        return "No balances found", 404
    balances_serialized = blockchain.serialize_balances(balances_raw).hex()
    return balances_serialized, 200


@app.route("/get-balances-hex-json", methods=["GET"])
@disallow_blacklisted
def get_balances_hex_json():
    balances_raw: Optional[dict[bytes, int]] = miner.get_balances()
    if balances_raw is None:
        return "No balances found", 404
    balances = {k.hex(): v for k, v in balances_raw.items()}
    return json.dumps(balances), 200


@app.route("/get-last-block-hex", methods=["GET"])
@disallow_blacklisted
def get_last_block_hex():
    chain: Optional[bytes] = miner.get_chain()
    if chain is None:
        return "No block found", 404
    block = chain[: blockchain.Block.N_BYTES]
    return block.hex(), 200


@app.route("/submit-transaction", methods=["POST"])
@disallow_blacklisted
def submit_transaction():
    transaction = (
        request.form["transaction"].encode("latin-1")
        if "transaction" in request.form
        else request.data
    )
    add_transaction(transaction)
    return "transaction submitted", 200


@app.route("/submit-chain", methods=["POST"])
@disallow_blacklisted
def submit_chain():
    chain = (
        request.form["chain"].encode("latin-1")
        if "chain" in request.form
        else request.data
    )
    add_chain(chain)
    return "chain submitted", 200


@app.route("/submit-block", methods=["POST"])
@disallow_blacklisted
def submit_block():
    block = (
        request.form["block"].encode("latin-1")
        if "block" in request.form
        else request.data
    )
    add_block(block)
    return "block submitted", 200


@app.route("/get-chain", methods=["GET"])
@disallow_blacklisted
def get_chain():
    chain: Optional[bytes] = miner.get_chain()
    if chain is None:
        return "No chain found", 404
    return chain.decode("latin-1"), 200


@app.route("/get-balances", methods=["GET"])
@disallow_blacklisted
def get_balances():
    balances_raw: Optional[dict[bytes, int]] = miner.get_balances()
    if balances_raw is None:
        return "No balances found", 404
    balances_serialized = blockchain.serialize_balances(balances_raw).decode("latin-1")
    return balances_serialized, 200


@app.route("/get-balances-json", methods=["GET"])
@disallow_blacklisted
def get_balances_json():
    balances_raw: Optional[dict[bytes, int]] = miner.get_balances()
    if balances_raw is None:
        return "No balances found", 404
    balances = {k.decode("latin-1"): v for k, v in balances_raw.items()}
    return json.dumps(balances), 200


@app.route("/get-last-block", methods=["GET"])
@disallow_blacklisted
def get_last_block():
    chain: Optional[bytes] = miner.get_chain()
    if chain is None:
        return "No block found", 404
    block = chain[: blockchain.Block.N_BYTES]
    return block.decode("latin-1"), 200


@app.route("/register-as-other-miner", methods=["GET", "POST"])
@disallow_blacklisted
def register_as_other_miner():
    if request.method == "GET":
        return render_template("register_as_other_miner.html"), 200
    if "miner_address" not in request.form:
        return "no miner address provided", 400
    miner_address = request.form["miner_address"]
    if miner_address not in other_miners:
        other_miners.add(miner_address)
    return "registered", 200


@app.route("/get-validity-hex", methods=["GET", "POST"])
@disallow_blacklisted
def get_validity_hex():
    if request.method == "GET":
        return render_template("get_validity.html"), 200
    h = bytes.fromhex(request.form.get("hash", ""))
    if not h:
        return "no hash provided", 400
    validity = _hash_to_validity.get(h)
    return "true" if validity else "unknown" if validity is None else "false", (
        201 if validity is None else 200
    )


@app.route("/get-validity", methods=["POST"])
@disallow_blacklisted
def get_validity():
    h = request.form.get("hash", "").encode("latin-1")
    if not h:
        return "no hash provided", 400
    validity = _hash_to_validity.get(h)
    return "true" if validity else "unknown" if validity is None else "false", (
        201 if validity is None else 200
    )


@app.route("/blacklisted", methods=["GET"])
def route_blacklisted():
    sender = request.remote_addr
    time_until_unblock = (
        TEMPORARY_BLACKLIST_TIME + _temporary_blacklist.get(sender, 0) - time.time()
    )
    h, m, s = (
        time_until_unblock // (24 * 60),
        time_until_unblock // 60 % (24 * 60),
        time_until_unblock % 60,
    )
    if time_until_unblock > 0:
        return f"Time until unblock: {h:02.0f}:{m:02.0f}:{s:02.0f}", 200
    else:
        return "You are unblocked", 200


@app.route("/get-known-miners", methods=["GET"])
def get_known_miners():
    return "\n".join(other_miners), 200


if __name__ == "__main__":
    # NOTE: importing blockchain here to prevent flask hot reloading from unloading the cuda miner
    import blockchain

    mp.freeze_support()
    app_pid = os.getpid()
    save_data_switch = threading.Event()
    save_data_switch.set()
    schedule(save_chain, CHAIN_SAVE_DELAY_SECS)
    schedule(save_known_miners, MINER_SAVE_DELAY_SECS)
    schedule(clean_caches, CACHE_CLEAN_DELAY_SECS)
    schedule(register_new_validities, REGISTER_VALIDITY_DELAY_SECS)
    app.secret_key = secrets.token_urlsafe(24)
    blockchain_config = blockchain.MiningConfig(
        collect_time=0,  # TODO make 60 when not testing
        broadcast_balances_as_bytes=False,
    )

    # shutdown password and salt generation
    shutdown_pw_salt = secrets.token_urlsafe(24)
    new_password = "".join(
        secrets.choice(string.ascii_letters + string.digits) for _ in range(64)
    )
    shutdown_pw_hash = hashlib.sha256(
        (new_password + shutdown_pw_salt).encode()
    ).hexdigest()
    print("The shutdown password is:", new_password)
    if os.path.exists("config.json"):
        with open("config.json", "r") as f:
            config = json.load(f)
    else:
        config = {}

    parser = argparse.ArgumentParser(description="Run a miner")
    parser.add_argument(
        "--port", type=int, default=5000, help="Port to run the miner on"
    )
    parser.add_argument("--cuda", action="store_true", help="Use CUDA for mining")
    args = parser.parse_args()
    if args.cuda:
        config["cuda"] = True
    if config.get("cuda", False):
        blockchain.schedule_jit_cuda_miner_load(
            "cuda_miner_templates/build/nonce128_intrinsic_uint128_random.ptx"
        )
    else:
        print(
            "Warning: Using slow debug Miner, use run with --cuda for GPU-powered mining or set cuda to true in config.json"
        )

    blockchain_config.version = config.get("version", 0)
    # load validator and blockchain from chain.bin file (if it exists) by deserializing the bytes
    chain_run_with = "mp"
    miner = None
    init_balance_hex_keys = config.get("init_balances", {})
    init_balances = {}
    for key, b in init_balance_hex_keys.items():
        init_balances[bytes.fromhex(key)] = b
    if os.path.exists("chains/chain.bin"):
        with open("chains/chain.bin", "rb") as f:
            miner = blockchain.Miner.from_bytes(
                f.read(),
                init_balances=init_balances,
                config=blockchain_config,
                run_with=chain_run_with,
                start_immediately=False,
            )
    elif os.path.exists("keys/public_key.bin"):
        with open("keys/public_key.bin", "rb") as f:
            miner = blockchain.Miner(
                f.read(),
                init_balances=init_balances,
                config=blockchain_config,
                run_with=chain_run_with,
                start_immediately=False,
            )
    else:
        private_key, public_key = blockchain.gen_key_pair()
        with open("keys/public_key.bin", "wb") as f:
            f.write(public_key)
        with open("keys/private_key.bin", "wb") as f:
            f.write(private_key)
        miner = blockchain.Miner(
            public_key,
            init_balances=init_balances,
            config=blockchain_config,
            run_with=chain_run_with,
            start_immediately=False,
        )

    if config.get("cuda", False):
        miner.use_cuda_miner()

    miner.start()
    app.run(port=args.port)
