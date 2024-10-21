import secrets
import requests
import threading
import math
import json
import time

from typing import Union, Optional
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


def run_sim():
    global next_market_step
    last_step_time = time.time()
    while True:
        t = time.time()
        ideal_batches = (t - last_step_time) * SIM_BATCHES_PER_SECOND
        n_batches = int(ideal_batches)
        with sim_data_lock:
            for _ in range(n_batches):
                for _ in range(SIM_BATCH_SIZE):
                    sim_data.append(sim.step())
                    while len(sim_data) > SIM_DATA_BACKLOG_SIZE:
                        sim_data.pop(0)
                    next_market_step += 1
        # only add the fraction of the difference that has actually resulted in steps to avoid skewing the tick rate
        if ideal_batches > 0:
            last_step_time += (t - last_step_time) * n_batches / ideal_batches


def user_get_available_points(user_id) -> int:
    resp = requests.get(
        f"{config['points_service']}/api/public/user/points/{user_id}",
        headers=api_headers,
    )
    return resp.json().get("points")


def withdraw_points(user_id, amount: int) -> dict:
    resp = requests.post(
        f"{config['points_service']}/api/public/user/withdraw",
        json.dumps({"userId": user_id, "points": amount}),
        headers=api_headers,
    )
    return resp.json()


def deposit_points(user_id, amount: int) -> dict:
    resp = requests.post(
        f"{config['points_service']}/api/public/user/deposit",
        json.dumps({"userId": user_id, "points": amount}),
        headers=api_headers,
    )
    return resp.json()


def jsonify_empty():
    return jsonify()


def jsonify_update(**kwargs):
    return jsonify(**kwargs)


def jsonify_error(err: str):
    return jsonify(error=err)


def jsonify_redirect(url: str):
    return jsonify(redirect=url)


def get_coins_to_points_exchange_rate() -> float:
    return sim.price


def get_points_to_coins_exchange_rate() -> float:
    return 1 / sim.price


def sim_data_to_json(
    data: list[tuple[int, tuple[str, str]]]
) -> list[dict[str, Union[int, Optional[dict[str, str]]]]]:
    return [
        {
            "exchange_rate": x[0],
            "event": (
                {"sentiment": x[1][0], "message": x[1][1]} if x[1] is not None else None
            ),
        }
        for x in data
    ]


class User(UserMixin):
    pass


@login_manager.user_loader
def load_user(user_id):
    if user_id in users:
        return users[user_id]
    user = User()
    user.id = user_id
    return users.setdefault(user_id, user)


@app.route("/login")
def login():
    state = secrets.token_urlsafe(16)
    redirect_uri = url_for("oauth_callback", _external=True)
    points_service_url = f"{config['points_service']}/api/auth/createRedirectUri"
    response = requests.get(
        points_service_url, params={"state": state, "redirect_uri": redirect_uri}
    )
    if response.status_code == 200:
        redirect_url = response.json().get("value")
        if not redirect_url:
            return "Error in authentication process: Your user id is invalid", 400
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


@app.route("/oauth/callback")
def oauth_callback():
    state = request.args.get("state")
    user_id = request.args.get("userId")
    if state and user_id:
        user = User()
        user.id = user_id
        users[user_id] = user
        login_user(user)
        return redirect(url_for("index")), 200
    return "Error in OAuth callback", 400


@app.route("/api/buy", methods=["POST"])
@login_required
def buy():
    user_id = current_user.id
    n_coins_bought = request.form.get("amount_coins_bought")
    if n_coins_bought is None or not n_coins_bought.isdigit() or int(n_coins_bought) <= 0:
        return jsonify_error("Invalid amount"), 400
    n_coins_bought = int(n_coins_bought)
    amount = n_coins_bought * get_coins_to_points_exchange_rate()
    user_amount_available = user_get_available_points(user_id)
    if user_amount_available is None:
        return jsonify_error("Unable to retrieve your point score"), 400
    if user_amount_available >= amount:
        user_info = user_management.get_user_info(user_id)
        try:
            cc_utils.send_coins(
                config["beff_jezos_user_info"], user_info.public_key, n_coins_bought
            )
        except Exception:
            return jsonify_error("Exception occured while processing request"), 400
        withdraw_points(
            user_id, math.ceil(n_coins_bought * get_coins_to_points_exchange_rate())
        )
        return (
            jsonify(
                coins_available=cc_utils.get_available_coins(user_info.public_key),
                n_coins_bought=n_coins_bought,
            ),
            200,
        )
    return jsonify_error("Too few points"), 400


@app.route("/api/sell", methods=["POST"])
@login_required
def sell():
    user_id = current_user.id
    amount_coins = request.form.get("amount_coins_sold")
    if amount_coins is None or not amount_coins.isdigit() or int(amount_coins) <= 0:
        return jsonify_error("Invalid amount"), 400
    amount_coins = int(amount_coins)
    user_info = user_management.get_user_info(user_id)
    user_amount_available = cc_utils.get_available_coins(user_info.public_key)
    if user_amount_available is None:
        return jsonify_error("Unable to retrieve your point score"), 400
    if user_amount_available >= amount_coins:
        n_points_to_recv = int(amount_coins * get_coins_to_points_exchange_rate())
        try:
            cc_utils.send_coins(
                user_info, config["beff_jezos_user_info"].public_key, amount_coins
            )
        except Exception:
            return jsonify_error("Exception occured while processing request"), 400
        deposit_points(user_id, n_points_to_recv)
        return (
            jsonify_update(
                coins_available=cc_utils.get_available_coins(user_info.public_key)
            ),
            200,
        )
    return jsonify_error("Too few coins"), 400


@app.route("/api/market/current-exchange-rate")
def api_route_get_coins_to_points_exchange_rate():
    with sim_data_lock:
        return (
            jsonify(
                exchange_rate=get_coins_to_points_exchange_rate(),
                next_market_step=next_market_step,
            ),
            200,
        )


@app.route("/api/market/latest-steps/<int:n>")
def api_route_get_latest_n_market_steps(n: int):
    with sim_data_lock:
        if n > SIM_DATA_BACKLOG_SIZE:
            return (
                jsonify_error(f"n exceeds backlog size of {SIM_DATA_BACKLOG_SIZE}"),
                400,
            )
        return (
            jsonify(
                market_steps=sim_data_to_json(sim_data[-n:]),
                n_retrieved=min(len(sim_data), n),
                next_market_step=next_market_step,
            ),
            200,
        )


@app.route("/api/market/steps-since/<int:n>")
def api_route_get_market_steps_since(n: int):
    with sim_data_lock:
        if n >= next_market_step:
            return (
                jsonify_error(
                    f"n exceeds or is equal to the index of the next market step ({next_market_step}), which has not been simulated yet"
                ),
                400,
            )
        return (
            jsonify(
                market_steps=sim_data_to_json(sim_data[n - next_market_step :]),
                n_retrieved=min(len(sim_data), n),
                next_market_step=next_market_step,
            ),
            200,
        )


@app.route("/api/market/next-market-step")
def api_route_get_next_market_step():
    return jsonify(next_market_step=next_market_step), 200


@app.route("/")
@login_required
def index():
    user_id = current_user.id
    info = user_management.get_user_info(user_id)
    return (
        render_template(
            "index.html", coins_available=cc_utils.get_available_coins(info.public_key)
        ),
        200,
    )


if __name__ == "__main__":
    import tdot_fake_market as market
    import cc_utils
    import user_management

    sim = market.MarketSim(
        base_value=30.0,
        bounce_back_value=10.0,
        market_boost_increase_factor=1.0,
        seed=42,
        stddev=0.04,
        freqs=[1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024],
        extra_freq_weights=[1, 1, 1, 1, 1, 1, 1, 1, 1 / 2, 1 / 4, 1 / 8],
        event_decay_factor=0.9999,
        event_impact=0.05,
        event_boost_weight=100,
        event_prob=0.01,
        event_change_trend_min_stds_from_mean=1,
        event_change_trend_weight=0.8,
    )
    SIM_BATCHES_PER_SECOND = 4  # num sim batches per second
    sim_data = []
    next_market_step = 0
    SIM_DATA_BACKLOG_SIZE = 50_000
    SIM_BATCH_SIZE = (
        10  # how many steps are computed in one go to avoid locking too often
    )
    users = {}
    sim_data_lock = threading.Lock()
    with open("config.json") as f:
        config = json.load(f)
    # convert the json repr of beff jezos into a CamlCoinUserInfo object
    bjzs = config["beff_jezos_user_info"]
    config["beff_jezos_user_info"] = user_management.CamlCoinUserInfo(
        bjzs["user_id"],
        int(bjzs["private_key"], base=16).to_bytes(
            cc_utils.PRIVATE_KEY_SIZE, cc_utils.ENDIAN
        ),
        int(bjzs["public_key"], base=16).to_bytes(
            cc_utils.PUBLIC_KEY_SIZE, cc_utils.ENDIAN
        ),
    )
    api_headers = {"Authorization": f"Bearer {config['api_key']}"}
    user_management.create_users_table()
    sim_thread = threading.Thread(target=run_sim)
    sim_thread.start()
    app.run(host="127.0.0.1", debug=True, use_reloader=False)
