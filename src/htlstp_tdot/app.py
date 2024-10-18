import secrets
import json

import cc_utils
import requests
from flask import Flask, url_for, redirect, request, render_template, jsonify
from flask_login import (
    LoginManager,
    UserMixin,
    login_user,
    logout_user,
    current_user,
    login_required,
)

app = Flask(__name__)

app.secret_key = secrets.token_urlsafe(24)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"
users = {}
with open("config.json") as f:
    config = json.load(f)
api_headers = {"Authentication": f"Bearer {config['api_key']}"}


def user_get_available_points(user_id):
    return (
        requests.get(
            f"{config['points_service']}/api/public/user/points/{user_id}",
            headers=api_headers,
        )
        .json()
        .get("points")
    )


def withdraw_points(user_id, amount: int):
    return requests.post(
        f"{config['points_service']}/api/public/user/withdraw",
        json.dumps({"userId": user_id, "points": amount}),
        headers=api_headers,
    )


def deposit_points(user_id, amount: int):
    return requests.post(
        f"{config['points_service']}/api/public/user/deposit",
        json.dumps({"userId": user_id, "points": amount}),
        headers=api_headers,
    )


def jsonify_empty():
    return jsonify()


def jsonify_update(**kwargs):
    return jsonify(**kwargs)


def jsonify_error(err: str):
    return jsonify(error=err)


def jsonify_redirect(url: str):
    return jsonify(redirect=url)


def get_user_info(user_id):
    # TODO make db to store the users secrets and retrieve the users cached num caml coins, have thread running in background reloading the actual
    # TODO balances from the blockchain
    return ()


def get_exchange_course() -> float:
    # TODO
    return 1.0


class User(UserMixin):
    def __init__(self, id, state):
        self.id = id
        self.state = state


@login_manager.user_loader
def load_user(user_id, user_state):
    if user_id in users:
        return users[user_id]
    return users.setdefault(User(user_id, user_state))


@app.route("/login")
def login():
    state = secrets.token_urlsafe(16)
    redirect_uri = url_for("oauth_callback", _external=True)
    points_service_url = f"{config['points_service']}/api/auth/createRedirectUri"
    response = requests.get(
        points_service_url, params={"state": state, "redirect_uri": redirect_uri}
    )
    if response.status_code == 200:
        redirect_url = response.json().get("redirect_uri")
        return redirect(redirect_url), 200
    return "Error in authentication process", 500


@app.route("/logout")
def logout():
    users.pop(current_user.id)
    logout_user()


@app.route("/oauth_callback")
def oauth_callback():
    state = request.args.get("state")
    user_id = request.args.get("userId")
    if state and user_id:
        user = User(user_id, state)
        login_user(user)
        return redirect(url_for("index")), 200
    return "Error in OAuth callback", 400


@app.route("/api/buy", methods=["POST"])
@login_required
def buy():
    user_id = current_user.id
    amount = request.form.get("amount_points_spent")
    if amount is None or not amount.isdigit() or int(amount) <= 0:
        return jsonify_error("Invalid amount"), 400
    user_amount_available = user_get_available_points(user_id)
    if user_amount_available is None:
        return jsonify_error("Unable to retrieve your point score"), 400
    if user_amount_available >= amount:
        user_info = get_user_info(user_id)
        try:
            cc_utils.send_coins(
                config["beff_jezos_user_info"], user_info.public_key, amount
            )
        except Exception:
            return jsonify_error("Exception occured while processing request"), 400
        withdraw_points(user_id, int(amount))
        return (
            jsonify_update(coins_available=cc_utils.get_available_coins(user_info)),
            200,
        )
    return jsonify_error("Too few points"), 400


@app.route("/api/sell", methods=["POST"])
@login_required
def sell():
    user_id = current_user.id
    amount = request.form.get("amount_coins_sold")
    if amount is None or not amount.isdigit() or int(amount) <= 0:
        return jsonify_error("Invalid amount"), 400
    user_info = get_user_info(user_id)
    user_amount_available = cc_utils.get_available_coins(user_info)
    if user_amount_available is None:
        return jsonify_error("Unable to retrieve your point score"), 400
    if user_amount_available >= amount:
        try:
            cc_utils.send_coins(
                user_info, config["beff_jezos_user_info"].public_key, amount
            )
        except Exception:
            return jsonify_error("Exception occured while processing request"), 400
        deposit_points(user_id, int(amount))
        return (
            jsonify_update(coins_available=cc_utils.get_available_coins(user_info)),
            200,
        )
    return jsonify_error("Too few coins"), 400


@app.route("/index")
def index():
    user_id = current_user.id
    info = get_user_info(user_id)
    return (
        render_template(
            "index.html", coins_available=cc_utils.get_available_coins(info)
        ),
        200,
    )


if __name__ == "__main__":
    app.run(debug=True)
