import os
import string
import json
import hashlib
import secrets
import blockchain
import multiprocessing as mp
from flask import Flask, request

app = Flask(__name__)


@app.route('/')
def index():  # put application's code here
    return 'Hello World!', 200


@app.route('/shutdown', methods=['POST'])
def shutdown():
    password = request.form.get('password')
    if password is None:
        return 'No password provided', 400
    pw_hash = hashlib.sha256((password + shutdown_pw_salt).encode()).hexdigest()
    if pw_hash != shutdown_pw_hash:
        return 'Invalid password', 403
    chain, _ = chain_handler.end()
    with open('chains/chain.bin', 'wb') as f:
        f.write(chain)
    return 'Server shut down', 200


@app.route('/transaction', methods=['POST'])
def submit_transaction():
    transaction = request.data
    chain_handler.add_transaction(transaction)
    return 'Transaction performed', 200


@app.route('/submit-chain', methods=['POST'])
def submit_chain():
    chain = request.data
    chain_handler.add_chain(chain)
    return 'Chain submitted', 200


@app.route('/get-chain', methods=['GET'])
def get_chain():
    chain = chain_handler.get_chain()
    if chain is None:
        return 'No chain found', 404
    return chain.to_bytes().decode("utf-8"), 200


if __name__ == '__main__':
    mp.freeze_support()
    app.secret_key = secrets.token_urlsafe(24)

    # shutdown password and salt generation
    if os.path.exists("config.json"):
        with open("config.json", "r") as f:
            config = json.load(f)
        new_password = "".join(secrets.choice(string.ascii_letters + string.digits) for _ in range(64))
        print("The shutdown password is:", new_password)
        shutdown_pw_hash = config.get("shutdown_pw_hash", hashlib.sha256(new_password.encode()).hexdigest())
        config["shutdown_pw_hash"] = shutdown_pw_hash
        with open("config.json", "w") as f:
            json.dump(config, f, indent=2)
    else:
        shutdown_pw_hash = "".join(secrets.choice(string.ascii_letters + string.digits) for _ in range(64))
        with open("config.json", "w") as f:
            json.dump({"shutdown_pw_hash": shutdown_pw_hash}, f, indent=2)

    shutdown_pw_salt = secrets.token_urlsafe(24)

    # load validator and blockchain from chain.bin file (if it exists) by deserializing the bytes
    chain_run_with = "mp"
    chain_handler = None
    init_balance_hex_keys = config.get("initial_balances", {})
    init_balance = {}
    for public_key, balance in init_balance_hex_keys.items():
        init_balance[bytes.fromhex(public_key)] = balance
    if os.path.exists('chains/chain.bin'):
        with open('chains/chain.bin', 'rb') as f:
            chain_handler = blockchain.BlockchainHandler.from_bytes(f.read(), run_with=chain_run_with)
    elif os.path.exists('keys/public_key.bin'):
        with open('keys/public_key.bin', 'rb') as f:
            chain_handler = blockchain.BlockchainHandler(f.read(), init_balance=init_balance, run_with=chain_run_with)
    else:
        private_key, public_key = blockchain.gen_key_pair()
        with open('keys/public_key.bin', 'wb') as f:
            f.write(public_key)
        with open('keys/private_key.bin', 'wb') as f:
            f.write(private_key)
        chain_handler = blockchain.BlockchainHandler(public_key, init_balance=init_balance, run_with=chain_run_with)

    app.run()
