import os
import secrets
import blockchain as blockchain_mod
import multiprocessing as mp
from flask import Flask
from flask_login import login_required, UserMixin, LoginManager, login_user, logout_user, current_user

app = Flask(__name__)
app.secret_key = secrets.token_urlsafe(24)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'


def blockchain_handler():
    # load validator and blockchain from chain.bin file (if it exists) by deserializing the bytes
    if os.path.exists('chain.bin'):
        with open('chain.bin', 'rb') as f:
            chain = blockchain_mod.Blockchain.from_bytes(f.read())
    elif os.path.exists('public_key.bin'):
        with open('public_key.bin', 'rb') as f:
            chain = blockchain_mod.Blockchain(f.read())
    else:
        private_key, public_key = blockchain_mod.gen_key_pair()
        with open('public_key.bin', 'wb') as f:
            f.write(public_key)
        with open('private_key.bin', 'wb') as f:
            f.write(private_key)
        chain = blockchain_mod.Blockchain(public_key)


@app.route('/')
def index():  # put application's code here
    return 'Hello World!'


@app.route('/shutdown', methods=['POST'])
def shutdown():
    shutdown_event.set()
    chain_handler.join()
    return 'Server shut down'


shutdown_event = mp.Event()
chain_handler = mp.Process(target=blockchain_handler, args=(shutdown_event,))


def main():
    chain_handler.start()
    app.run()


if __name__ == '__main__':
    main()
