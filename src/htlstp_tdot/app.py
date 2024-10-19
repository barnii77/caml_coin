import secrets
import threading
import math
import json
import tdot_fake_market as market
import cc_utils
import requests
import user_management
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
sim = market.MarketSim(
    market_boost_increase_factor=1.0,
    seed=42,
    stddev=0.04,
    freqs=[1, 2, 4, 8, 32, 64, 128, 256, 512],
    decay_factor=0.96,
    event_impact=1.25,
    event_prob=0.02,
)
sim_data = []
SIM_DATA_BACKLOG_SIZE = 50_000
users = {}
with open("config.json") as f:
    config = json.load(f)
# convert the json repr of beff jezos into a CamlCoinUserInfo object
bjzs = config["beff_jezos_user_info"]
config["beff_jezos_user_info"] = user_management.CamlCoinUserInfo(
    bjzs["user_id"], bjzs["private_key"], bjzs["public_key"]
)
api_headers = {"Authentication": f"Bearer {config['api_key']}"}


def run_sim():
    while True:
        sim_data.append(sim.step())
        while len(sim_data) > SIM_DATA_BACKLOG_SIZE:
            sim_data.pop(0)


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


def get_coins_to_points_exchange_course() -> float:
    return sim.price


def get_points_to_coins_exchange_course() -> float:
    return 1 / sim.price


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
@login_required
def logout():
    users.pop(current_user.id)
    logout_user()
    return redirect(url_for("logged_out")), 200


@app.route("/logged-out")
def logged_out():
    return render_template("logged_out.html"), 200


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
        user_info = user_management.get_user_info(user_id)
        n_coins_bought = int(int(amount) * get_points_to_coins_exchange_course())
        try:
            cc_utils.send_coins(
                config["beff_jezos_user_info"], user_info.public_key, n_coins_bought
            )
        except Exception:
            return jsonify_error("Exception occured while processing request"), 400
        withdraw_points(
            user_id, math.ceil(n_coins_bought * get_coins_to_points_exchange_course())
        )
        return (
            jsonify(
                coins_available=cc_utils.get_available_coins(user_info),
                n_coins_bought=n_coins_bought,
            ),
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
    user_info = user_management.get_user_info(user_id)
    user_amount_available = cc_utils.get_available_coins(user_info)
    if user_amount_available is None:
        return jsonify_error("Unable to retrieve your point score"), 400
    if user_amount_available >= amount:
        n_points_to_recv = int(int(amount) * get_coins_to_points_exchange_course())
        try:
            cc_utils.send_coins(
                user_info, config["beff_jezos_user_info"].public_key, int(amount)
            )
        except Exception:
            return jsonify_error("Exception occured while processing request"), 400
        deposit_points(user_id, n_points_to_recv)
        return (
            jsonify_update(coins_available=cc_utils.get_available_coins(user_info)),
            200,
        )
    return jsonify_error("Too few coins"), 400


@app.route("/api/exchange-course/current")
def get_coins_to_points_exchange_course():
    return jsonify(exchange_course=get_coins_to_points_exchange_course()), 200


@app.route("/api/exchange-course/latest/<int:n>")
def get_latest_n_coins_to_points_exchange_courses(n: int):
    if n > SIM_DATA_BACKLOG_SIZE:
        return jsonify_error(f"n exceeds backlog size of {SIM_DATA_BACKLOG_SIZE}"), 400
    return jsonify(exchange_courses=sim_data[-n:]), 200


@app.route("/")
def index():
    if not current_user.is_authenticated:
        return redirect(url_for("login")), 200
    user_id = current_user.id
    info = user_management.get_user_info(user_id)
    return (
        render_template(
            "index.html", coins_available=cc_utils.get_available_coins(info)
        ),
        200,
    )


if __name__ == "__main__":
    user_management.create_users_table()
    sim_thread = threading.Thread(target=run_sim)
    app.run(debug=True)
