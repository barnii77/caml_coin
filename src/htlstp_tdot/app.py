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


# TODO this function must constantly request a users fake points and if fake_points <= 0, request the real score
# TODO so it can auto-sell once the user is low on points (low on points = not enough money to afford a single coin?)
def run_broker():
    while True:
        t = time.time()
        for user_id, (
            leverage,
            open_exchange_rate,
            open_time,
        ) in broker_positions.items():
            if t - open_time >= BROKER_MAX_HOLD_TIME_SECS:
                user_close_position(user_management.get_user_info(user_id))


def user_close_position(user_info) -> int:
    leverage, starting_exchange_rate, open_time = broker_positions.pop(
        user_info.user_id
    )
    holding_time = time.time() - open_time
    leverage_normalized_fees = (
        holding_time // BROKER_KEEP_FEE_INTERVAL_SECS
    ) * BROKER_KEEP_FEE + BROKER_OPEN_FEE
    net_earnings = round(
        leverage
        * (
            get_coins_to_points_exchange_rate()
            - starting_exchange_rate
            - leverage_normalized_fees
        )
    )
    if net_earnings > 0:
        deposit_points(user_info, net_earnings)
    elif net_earnings < 0:
        withdraw_points(user_info, -net_earnings)
    return net_earnings


def user_get_fake_points(user_info) -> int:
    return (
        max(MAX_PROFIT_POINTS, cc_utils.get_available_fake_points(user_info.public_key))
        - MAX_PROFIT_POINTS
    )


def user_get_available_points(user_info) -> int:
    # have to do this sub because first MAX_PROFIT_POINTS fake points are synced with points service
    fake_points = (
        max(MAX_PROFIT_POINTS, cc_utils.get_available_fake_points(user_info.public_key))
        - MAX_PROFIT_POINTS
    )
    resp = requests.get(
        f"{config['points_service']}/api/public/user/points/{user_info.user_id}",
        headers=api_headers,
    )
    return fake_points + resp.json().get("points")


def withdraw_points(user_info, amount: int):
    new_fake_points = cc_utils.withdraw_fake_points(user_info.public_key, amount)
    delta_real = max(amount - max(new_fake_points + amount - MAX_PROFIT_POINTS, 0), 0)
    if delta_real:
        requests.post(
            f"{config['points_service']}/api/public/user/withdraw",
            json={"userId": user_info.user_id, "points": delta_real},
            headers=api_headers,
        )


def deposit_points(user_info, amount: int):
    new_fake_points = cc_utils.deposit_fake_points(user_info.public_key, amount)
    delta_real = max(amount - max(new_fake_points - MAX_PROFIT_POINTS, 0), 0)
    if delta_real:
        requests.post(
            f"{config['points_service']}/api/public/user/deposit",
            json={"userId": user_info.user_id, "points": delta_real},
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


def requested_leverage_allowed_with_points(
    user_real_points, leverage: int
) -> tuple[bool, int]:
    for lev, min_fake_points_req in broker_leverage_min_points_pairs:
        if leverage >= lev and user_real_points < min_fake_points_req:
            return False, min_fake_points_req
    return True, -1


def get_user_id_from_login_token(login_token) -> str:
    resp = requests.get(
        f"{config['points_service']}/api/auth/user?token={login_token}",
        headers=api_headers,
    )
    return resp.text[1:-1]


def is_valid_user_id(user_id) -> bool:
    resp = requests.get(
        f"{config['points_service']}/api/public/user/points/{user_id}",
        headers=api_headers,
    )
    try:
        j = resp.json()
    except requests.exceptions.JSONDecodeError:
        return False
    else:
        return "points" in j


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
        points_service_url,
        params={"state": state, "redirect_uri": redirect_uri},
        headers=api_headers,
    )
    if response.status_code == 200:
        redirect_url = response.json().get("value")
        if not redirect_url:
            return "Error in authentication process: Your user id is invalid", 400
        return redirect(redirect_url), 200
    return "Error in authentication process", 500


@app.route("/manual-login", methods=["GET", "POST"])
def manual_login():
    if request.method == "GET":
        return render_template("manual_login.html"), 200
    user_id = request.form.get("userId")
    login_token = request.form.get("token")
    if user_id and is_valid_user_id(user_id) or login_token:
        if login_token:
            user_id = get_user_id_from_login_token(login_token)
            if not user_id:
                return "Invalid token", 400
        user = User()
        user.id = user_id
        users[user_id] = user
        login_user(user)
        return redirect(url_for("index")), 200

    return "Invalid user id", 400


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
    user_id = request.args.get("userId")
    if user_id:
        user = User()
        user.id = user_id
        users[user_id] = user
        login_user(user)
        return redirect(url_for("index")), 200
    return "Error in OAuth callback", 400


@app.route("/api/get-scores")
@login_required
def api_route_get_scores():
    user_id = current_user.id
    user_info = user_management.get_user_info(user_id)
    return jsonify(
        coins_available=cc_utils.get_available_coins(user_info.public_key),
        fake_points=cc_utils.get_available_fake_points(user_info.public_key),
    )


@app.route("/api/buy", methods=["POST"])
@login_required
def api_route_buy():
    user_id = current_user.id
    n_coins_bought = request.form.get("amount_coins_bought")
    if (
        n_coins_bought is None
        or not n_coins_bought.isdigit()
        or int(n_coins_bought) <= 0
    ):
        return jsonify_error("Invalid amount"), 400
    n_coins_bought = int(n_coins_bought)
    amount = n_coins_bought * get_coins_to_points_exchange_rate()
    user_info = user_management.get_user_info(user_id)
    user_amount_available = user_get_available_points(user_info)
    if user_amount_available is None:
        return jsonify_error("Unable to retrieve your point score"), 400
    if amount > user_amount_available:
        amount = user_amount_available
    if amount > 0:
        try:
            cc_utils.send_coins(
                config["beff_jezos_user_info"], user_info.public_key, n_coins_bought
            )
        except Exception as e:
            return (
                jsonify_error("Exception occured while processing request: " + str(e)),
                400,
            )
        withdraw_points(
            user_info, math.ceil(n_coins_bought * get_coins_to_points_exchange_rate())
        )
        return (
            jsonify(
                coins_available=cc_utils.get_available_coins(user_info.public_key),
                n_coins_bought=n_coins_bought,
                fake_points=cc_utils.get_available_fake_points(user_info.public_key),
            ),
            200,
        )
    return jsonify_error("Too few points"), 400


@app.route("/api/sell", methods=["POST"])
@login_required
def api_route_sell():
    user_id = current_user.id
    amount_coins = request.form.get("amount_coins_sold")
    if amount_coins is None or not amount_coins.isdigit() or int(amount_coins) <= 0:
        return jsonify_error("Invalid amount"), 400
    amount_coins = int(amount_coins)
    user_info = user_management.get_user_info(user_id)
    user_amount_available = cc_utils.get_available_coins(user_info.public_key)
    if user_amount_available is None:
        return jsonify_error("Unable to retrieve your point score"), 400
    if amount_coins > user_amount_available:
        amount_coins = user_amount_available
    if amount_coins > 0:
        n_points_to_recv = int(amount_coins * get_coins_to_points_exchange_rate())
        try:
            cc_utils.send_coins(
                user_info, config["beff_jezos_user_info"].public_key, amount_coins
            )
        except Exception as e:
            return (
                jsonify_error("Exception occured while processing request: " + str(e)),
                400,
            )
        deposit_points(user_info, n_points_to_recv)
        return (
            jsonify_update(
                coins_available=cc_utils.get_available_coins(user_info.public_key),
                n_coins_sold=amount_coins,
                fake_points=cc_utils.get_available_fake_points(user_info.public_key),
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


@app.route("/api/broker/open-position", methods=["POST"])
@login_required
def api_route_broker_open_position():
    user_id = current_user.id
    if user_id in broker_positions:
        return jsonify_error("User already has open position"), 400

    leverage = request.form.get("leverage")
    if leverage is None or not leverage.isdigit() or int(leverage) <= 0:
        return jsonify_error("Invalid leverage"), 400

    leverage = int(leverage)
    user_info = user_management.get_user_info(user_id)
    real_points = user_get_available_points(user_info)
    leverage_is_allowed, min_points_req_for_leverage = (
        requested_leverage_allowed_with_points(real_points, leverage)
    )
    if not leverage_is_allowed:
        return jsonify_error(
            f"For this amount of leverage, you need at least {min_points_req_for_leverage}"
        )

    broker_positions[user_id] = (
        leverage,
        get_coins_to_points_exchange_rate(),
        time.time(),
    )
    return (
        jsonify(
            open_fee=BROKER_OPEN_FEE,
            keep_fee=BROKER_KEEP_FEE,
            keep_fee_interval=BROKER_KEEP_FEE_INTERVAL_SECS,
        ),
        200,
    )


@app.route("/api/broker/close-position", methods=["POST"])
@login_required
def api_route_broker_close_position():
    user_id = current_user.id
    if user_id not in broker_positions:
        return jsonify_error("User does not have an open position"), 400

    user_info = user_management.get_user_info(user_id)
    net_earnings = user_close_position(user_info)
    return (
        jsonify(net_earnings=net_earnings),
        200,
    )


@app.route("/api/broker/get-position")
@login_required
def api_route_broker_get_position():
    user_id = current_user.id
    if user_id in broker_positions:
        lev, open_xcr, open_time = broker_positions[user_id]
        return jsonify(leverage=lev, open_exchange_rate=open_xcr, open_time=open_time)
    else:
        return jsonify_error("no open position")


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

    broker_positions = {}
    broker_leverage_min_points_pairs = [(150, 2000), (30, 1000)]
    BROKER_OPEN_FEE = 0.05  # amount of points it costs to open a position at leverage=1
    BROKER_KEEP_FEE = (
        0.2  # amount of points it costs to keep an open position at leverage=1
    )
    BROKER_KEEP_FEE_INTERVAL_SECS = 10  # interval in which BROKER_KEEP_FEE is charged
    BROKER_MAX_HOLD_TIME_SECS = 120  # after this amount of time, auto-sell (so users can't forget about their position and loose everything to keep fees)

    SIM_BATCHES_PER_SECOND = 3  # num sim batches per second
    SIM_BATCH_SIZE = (
        3  # how many steps are computed in one go to avoid locking too often
    )
    seed = 44
    sim = market.MarketSimMix(
        market.MarketSim(  # this sim provides high frequency "day trading" dynamics that can be changed by fake X posts etc
            base_value=100.0,
            bounce_back_value=20.0,
            market_boost_increase_factor=1.0,
            seed=seed,
            stddev=0.005,
            freqs=[
                1,
                2,
                4,
                8,
                16,
                32,
                64,
                128,
                256,
                512,
                1024,
            ],
            extra_freq_weights=[
                8,
                8,
                8,
                4,
                4,
                4,
                2,
                1.5,
                0.6,
                0.25,
                0.125,
            ],
            event_decay_factor=0.99999,
            event_impact=0.00125,
            event_boost_weight=50,
            event_prob=0.02,
            event_change_trend_min_stds_from_mean=1.0,
            event_change_trend_weight=0.9,
        ),
        market.MarketSim(  # this simulation provides low frequency hourly / 20-minute trends that must not be influenced by events
            base_value=0,
            bounce_back_value=-9999999999999999,
            market_boost_increase_factor=1.0,
            seed=seed,
            stddev=0.06,
            freqs=[
                20 * 15 * SIM_BATCH_SIZE * SIM_BATCHES_PER_SECOND,
                60 * 15 * SIM_BATCH_SIZE * SIM_BATCHES_PER_SECOND,
            ],
            extra_freq_weights=[
                96 / (20 * 20 * SIM_BATCHES_PER_SECOND * SIM_BATCH_SIZE),
                256 / (60 * 20 * SIM_BATCHES_PER_SECOND * SIM_BATCH_SIZE),
            ],
            event_decay_factor=0,
            event_impact=0,
            event_boost_weight=0,
            event_prob=0,
            event_change_trend_min_stds_from_mean=0,
            event_change_trend_weight=0,
        ),
    )
    MAX_PROFIT_POINTS = 500
    sim_data = []
    next_market_step = 0
    SIM_DATA_BACKLOG_SIZE = 50_000
    users = {}
    sim_data_lock = threading.Lock()
    with open("config.json") as f:
        config = json.load(f)
    # convert the json repr of beff jezos into a CamlCoinUserInfo object
    _beff_jezos: dict = config["beff_jezos_user_info"]
    config["beff_jezos_user_info"] = user_management.CamlCoinUserInfo(
        _beff_jezos["user_id"],
        int(_beff_jezos["private_key"], base=16).to_bytes(
            cc_utils.PRIVATE_KEY_SIZE, cc_utils.ENDIAN
        ),
        int(_beff_jezos["public_key"], base=16).to_bytes(
            cc_utils.PUBLIC_KEY_SIZE, cc_utils.ENDIAN
        ),
    )
    beff_jezos: "user_management.CamlCoinUserInfo" = config["beff_jezos_user_info"]
    cc_utils.balances[beff_jezos.public_key] = config["beff_jezos_wealth"]
    api_headers = {"Authorization": f"Bearer {config['api_key']}"}
    user_management.create_users_table()
    sim_thread = threading.Thread(target=run_sim)
    broker_thread = threading.Thread(target=run_broker)
    sim_thread.start()
    broker_thread.start()
    app.run(host="127.0.0.1", debug=True, use_reloader=False)
