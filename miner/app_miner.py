import os
import string
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

import blockchain

_chain_time_cache = {}
_transaction_time_cache = {}
_miner_n_non_responsive = {}
KEEP_IN_CACHE_SECS = 60
CACHE_CLEAN_DELAY_SECS = 20
BROADCAST_THREAD_POOL_SIZE = 10
OTHER_MINER_CONSIDER_INVALID_NUM_UNRESPONSIVE = 5

with open("known_miners.txt", "r") as f:
    other_miners = set(l.rstrip() for l in f.readlines())


def schedule(task, delay, num_repeats=-1):
    """
    Runs task num_repeats times (or indefinitely if num_repeats < 0), waiting for delay seconds before every execution.
    """

    def task_wrapper():
        n = num_repeats
        while n != 0:
            time.sleep(delay)
            task()
            n -= 1

    threading.Thread(target=task_wrapper, daemon=True).start()


def save_data():
    if not save_chain_switch.is_set():
        return
    chain = miner.get_chain()
    if chain is not None:
        with open('chains/chain.bin', 'wb') as f:
            f.write(chain)
    with open("known_miners.txt", "w") as f:
        f.write("\n".join(other_miners))


def _broadcast_to_miner(m: str, data: bytes, route: str):
    print("attempt to broadcast to", m, "data:", data.hex()[:20] + "...", "route:", route)
    time.sleep(1)
    try:
        requests.post(f"http://{m}/{route}", data=data)
    except requests.exceptions.RequestException:
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
    broadcaster_thread_pool = [
        threading.Thread(target=lambda miners: [_broadcast_to_miner(m, data, route) for m in miners],
                         args=(miner_split,), daemon=True)
        for miner_split in split_set_into_n_parts(other_miners, BROADCAST_THREAD_POOL_SIZE)]
    for t in broadcaster_thread_pool:
        t.start()
    for t in broadcaster_thread_pool:
        t.join()
    # for m in other_miners:
    #     _broadcast_to_miner(m, data, route)


def clean_caches():
    now = time.time()
    remove_keys = []
    for chain, t in _chain_time_cache.items():
        if now - t > KEEP_IN_CACHE_SECS:
            remove_keys.append(chain)
    for item in remove_keys:
        _chain_time_cache.pop(item)
    remove_keys.clear()
    for transaction, t in _transaction_time_cache.items():
        if now - t > KEEP_IN_CACHE_SECS:
            remove_keys.append(transaction)
    for item in remove_keys:
        _transaction_time_cache.pop(item)
    remove_keys.clear()
    for other_miner, c in _miner_n_non_responsive.items():
        if c >= OTHER_MINER_CONSIDER_INVALID_NUM_UNRESPONSIVE:
            other_miners.remove(other_miner)
            remove_keys.append(other_miner)
    for item in remove_keys:
        _miner_n_non_responsive.pop(item)


def add_chain(chain: bytes):
    if chain not in _chain_time_cache:
        miner.add_chain(chain)
        _chain_time_cache[chain] = time.time()
        broadcast_to_miners(chain, "submit-chain")


def add_transaction(transaction: bytes):
    if transaction not in _transaction_time_cache:
        miner.add_transaction(transaction)
        _transaction_time_cache[transaction] = time.time()
        broadcast_to_miners(transaction, "submit-transaction")


app = Flask(__name__)


@app.route('/')
def index():  # put application's code here
    return render_template("index.html", validator=miner.validator.hex(), len=len,
                           VALIDATOR_CHARS_PER_LINE=50), 200


@app.route('/shutdown', methods=['GET', 'POST'])
def shutdown():
    if request.method == 'GET':
        return render_template("shutdown.html"), 200
    else:
        password = request.form.get('password', request.data.decode("utf-8"))
        pw_hash = hashlib.sha256((password + shutdown_pw_salt).encode()).hexdigest()
        if pw_hash != shutdown_pw_hash:
            return redirect("/shutdown"), 403
        save_chain_switch.clear()  # avoid race condition with automatic chain saving
        chain, _ = miner.finish()
        if chain is not None:
            with open('chains/chain.bin', 'wb') as f:
                f.write(chain)

        schedule(lambda: os.kill(app_pid, signal.SIGTERM), 5, 1)
        return '<h1>Server shutting down</h1>', 200


@app.route('/submit-transaction-hex', methods=['GET', 'POST'])
def submit_transaction_hex():
    if request.method == 'GET':
        # render an html template that lets the user submit a transaction in hex
        return render_template("submit_transaction.html"), 200
    else:
        transaction = bytes.fromhex(request.form['transaction']) if 'transaction' in request.form else bytes.fromhex(
            str(request.data))
        add_transaction(transaction)
        return redirect(url_for('index')), 200


@app.route('/submit-chain-hex', methods=['GET', 'POST'])
def submit_chain_hex():
    if request.method == 'GET':
        return render_template("submit_chain.html"), 200
    else:
        chain = bytes.fromhex(request.form['chain']) if 'chain' in request.form else bytes.fromhex(str(request.data))
        add_chain(chain)
        return redirect(url_for('index')), 200


@app.route('/get-chain-hex', methods=['GET'])
def get_chain_hex():
    chain: Optional[bytes] = miner.get_chain()
    if chain is None:
        return 'No chain found', 404
    return chain.hex(), 200


@app.route('/get-balances-hex', methods=['GET'])
def get_balances_hex():
    balances_raw: Optional[dict[bytes, int]] = miner.get_balances()
    if balances_raw is None:
        return 'No balances found', 404
    balances_serialized = blockchain.serialize_balances(balances_raw).hex()
    return balances_serialized, 200


@app.route('/get-balances-hex-json', methods=['GET'])
def get_balances_hex_json():
    balances_raw: Optional[dict[bytes, int]] = miner.get_balances()
    if balances_raw is None:
        return 'No balances found', 404
    balances = {k.hex(): v for k, v in balances_raw.items()}
    return json.dumps(balances), 200


@app.route('/submit-transaction', methods=['POST'])
def submit_transaction():
    transaction = request.form['transaction'].encode("latin-1") if 'transaction' in request.form else request.data
    add_transaction(transaction)
    return "transaction submitted", 200


@app.route('/submit-chain', methods=['POST'])
def submit_chain():
    chain = request.form['chain'].encode("latin-1") if 'chain' in request.form else request.data
    add_chain(chain)
    return "chain submitted", 200


@app.route('/get-chain', methods=['GET'])
def get_chain():
    chain: Optional[bytes] = miner.get_chain()
    if chain is None:
        return 'No chain found', 404
    return chain.decode("latin-1"), 200


@app.route('/get-balances', methods=['GET'])
def get_balances():
    balances_raw: Optional[dict[bytes, int]] = miner.get_balances()
    if balances_raw is None:
        return 'No balances found', 404
    balances_serialized = blockchain.serialize_balances(balances_raw).decode("latin-1")
    return balances_serialized, 200


@app.route('/get-balances-json', methods=['GET'])
def get_balances_json():
    balances_raw: Optional[dict[bytes, int]] = miner.get_balances()
    if balances_raw is None:
        return 'No balances found', 404
    balances = {k.decode("latin-1"): v for k, v in balances_raw.items()}
    return json.dumps(balances), 200


@app.route("/register-as-other-miner", methods=['GET', 'POST'])
def register_as_other_miner():
    if request.method == 'GET':
        return render_template("register_as_other_miner.html"), 200
    if "miner_address" not in request.form:
        return "no miner address provided", 400
    miner_address = request.form["miner_address"]
    if miner_address not in other_miners:
        other_miners.add(miner_address)
    print("Registered", miner_address)
    return "registered", 200


if __name__ == '__main__':
    CHAIN_SAVE_DELAY_SECS = 2 * 60
    mp.freeze_support()
    app_pid = os.getpid()
    save_chain_switch = threading.Event()
    # save_chain_switch.set()
    schedule(save_data, CACHE_CLEAN_DELAY_SECS)
    schedule(clean_caches, KEEP_IN_CACHE_SECS)
    app.secret_key = secrets.token_urlsafe(24)
    blockchain_config = blockchain.MiningConfig(incompatible_chain_distrust=1, broadcast_balances_as_bytes=False)

    # shutdown password and salt generation
    shutdown_pw_salt = secrets.token_urlsafe(24)
    new_password = "".join(secrets.choice(string.ascii_letters + string.digits) for _ in range(64))
    shutdown_pw_hash = hashlib.sha256((new_password + shutdown_pw_salt).encode()).hexdigest()
    print("The shutdown password is:", new_password)
    if os.path.exists("config.json"):
        with open("config.json", "r") as f:
            config = json.load(f)
    else:
        config = {}

    # load validator and blockchain from chain.bin file (if it exists) by deserializing the bytes
    chain_run_with = "mp"
    miner = None
    init_balance_hex_keys = config.get("init_balance", {})
    init_balance = {}
    for key, b in init_balance_hex_keys.items():
        init_balance[bytes.fromhex(key)] = b
    if os.path.exists('chains/chain.bin'):
        with open('chains/chain.bin', 'rb') as f:
            miner = blockchain.Miner.from_bytes(f.read(), init_balance=init_balance,
                                                config=blockchain_config, run_with=chain_run_with)
    elif os.path.exists('keys/public_key.bin'):
        with open('keys/public_key.bin', 'rb') as f:
            miner = blockchain.Miner(f.read(), init_balance=init_balance, config=blockchain_config,
                                     run_with=chain_run_with)
    else:
        private_key, public_key = blockchain.gen_key_pair()
        with open('keys/public_key.bin', 'wb') as f:
            f.write(public_key)
        with open('keys/private_key.bin', 'wb') as f:
            f.write(private_key)
        miner = blockchain.Miner(public_key, init_balance=init_balance, config=blockchain_config,
                                 run_with=chain_run_with)

    parser = argparse.ArgumentParser(description='Run a miner')
    parser.add_argument('--port', type=int, default=5000, help='Port to run the miner on')
    args = parser.parse_args()
    app.run(port=args.port)
