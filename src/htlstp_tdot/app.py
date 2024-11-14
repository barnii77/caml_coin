import traceback
import secrets
import requests
import threading
import math
import json
import time
import hashlib
import functools

from typing import Union, Optional
from flask import (
    Flask,
    url_for,
    redirect,
    request,
    render_template,
    jsonify,
    send_from_directory,
)
from flask_login import (
    LoginManager,
    UserMixin,
    login_user,
    logout_user,
    current_user,
    login_required,
)
import tdot_fake_market as market
import cc_utils
import user_management

app = Flask(__name__, static_folder="__this_folder_does_not_exist__")

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
                next_market_step += SIM_BATCH_SIZE
        # only add the fraction of the difference that has actually resulted in steps to avoid skewing the tick rate
        if ideal_batches > 0:
            last_step_time += (t - last_step_time) * n_batches / ideal_batches


def run_background_services():
    while True:
        # state cleaning
        t = time.time()
        kv = list(states.items())
        for state, ts in kv:
            if ts - t > STATE_TIMEOUT:
                states.pop(state)

        # broker auto-functions like selling if the user will run out of money soon or applying open position timeouts
        t = time.time()
        kv = list(broker_positions.items())
        for user_id, (
            leverage,
            open_exchange_rate,
            open_time,
        ) in kv:
            user_info = user_management.get_user_info(user_id)
            gain = leverage * (get_coins_to_points_exchange_rate() - open_exchange_rate)
            a, b = get_min_backup_capital_for_leverage(leverage)
            min_required_backup_capital = a * get_coins_to_points_exchange_rate() + b

            if t - open_time >= BROKER_MAX_HOLD_TIME_SECS:
                user_close_position(user_management.get_user_info(user_id))
                if user_id not in user_broker_notifications:
                    user_broker_notifications[user_id] = []
                user_broker_notifications[user_id].append(
                    "Broker auto-closed your position because the time limit for open positions has been exceeded"
                )
            elif (
                user_get_fake_points(user_info.public_key) + gain
                <= min_required_backup_capital
                and user_get_available_points(user_info) + gain
                < min_required_backup_capital
            ):
                user_close_position(user_management.get_user_info(user_id))
                if user_id not in user_broker_notifications:
                    user_broker_notifications[user_id] = []
                user_broker_notifications[user_id].append(
                    f"Broker auto-closed your position because you dropped too low in backup capital for your selected leverage (you must have least {min_required_backup_capital} points)"
                )
            elif (
                open_exchange_rate >= get_coins_to_points_exchange_rate() * 2
            ):  # if user made more than 50% loss, auto-sell (eu broker regulation)
                user_close_position(user_management.get_user_info(user_id))
                if user_id not in user_broker_notifications:
                    user_broker_notifications[user_id] = []
                user_broker_notifications[user_id].append(
                    "Broker auto-closed your position because you have made a 50% or more loss and EU broker regulations require CamlCoin Broker AG to close your position."
                )

        # reload the config in case anything has changed
        try:
            with open("config.json") as f:
                new_config = json.load(f)
            for k, v in new_config.items():
                config[k] = v
        except Exception:
            traceback.format_exc()

        with open("data/cc_utils_balances.json", "w") as f:
            json.dump(cc_utils.balances, f)
        with open("data/cc_utils_fake_points.json", "w") as f:
            json.dump(cc_utils.fake_points, f)


def get_readable_name_of_user(user_id):
    if user_id not in user_id_to_readable_name:
        name = (
            requests.get(
                f"{config['points_service']}/api/public/user/name/{user_id}",
                headers=api_headers,
            )
            .json()
            .get("value")
        )
        user_id_to_readable_name[user_id] = name
    return user_id_to_readable_name[user_id]


def restart_on_crash(func):
    def wrapper(*args, **kwargs):
        while True:
            try:
                func(*args, **kwargs)
            except Exception:
                traceback.format_exc()

    return wrapper


def get_form_data():
    if not request.form:
        return request.json
    return request.form


def get_args_data():
    if not request.args:
        return request.json
    return request.args


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


def user_get_fake_points(public_key) -> int:
    # have to do this sub because first MAX_PROFIT_POINTS fake points are synced with points service
    return (
        0
        if MAX_PROFIT_POINTS is None
        else (
            max(MAX_PROFIT_POINTS, cc_utils.get_available_fake_points(public_key))
            - MAX_PROFIT_POINTS
        )
    )


def user_get_available_points(user_info) -> int:
    fake_points = user_get_fake_points(user_info.public_key)
    resp = requests.get(
        f"{config['points_service']}/api/public/user/points/{user_info.user_id}",
        headers=api_headers,
    )
    return fake_points + resp.json().get("points")


def withdraw_points(user_info, amount: int):
    new_fake_points = cc_utils.withdraw_fake_points(user_info.public_key, amount)
    delta_real = (
        amount
        if MAX_PROFIT_POINTS is None
        else max(amount - max(new_fake_points + amount - MAX_PROFIT_POINTS, 0), 0)
    )
    if delta_real:
        requests.post(
            f"{config['points_service']}/api/public/user/withdraw",
            json={"userId": user_info.user_id, "points": delta_real},
            headers=api_headers,
        )


def deposit_points(user_info, amount: int):
    new_fake_points = cc_utils.deposit_fake_points(user_info.public_key, amount)
    delta_real = (
        amount
        if MAX_PROFIT_POINTS is None
        else max(amount - max(new_fake_points - MAX_PROFIT_POINTS, 0), 0)
    )
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
    a, b = get_min_backup_capital_for_leverage(leverage)
    min_fake_points_req = math.ceil(get_coins_to_points_exchange_rate() * a + b)
    if user_real_points < min_fake_points_req:
        return False, min_fake_points_req
    return True, -1


def get_min_backup_capital_for_leverage(leverage: int):
    for lev, setting in broker_leverage_min_backup_capital_pairs:
        if leverage >= lev:
            return setting
    raise RuntimeError(
        "invalid configuration of broker_leverage_min_backup_capital_pairs or invalid value of leverage"
    )


def get_user_id_from_login_token(login_token) -> Optional[str]:
    resp = requests.get(
        f"{config['points_service']}/api/auth/user?token={login_token}",
        headers=api_headers,
    )
    try:
        j = resp.json()
        if "error" in j:
            return j
        return j.get("value")
    except requests.exceptions.JSONDecodeError:
        return None


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


def admin_route(method="POST"):
    method = method.lower()
    if method not in ("get", "post"):
        raise ValueError("unsupported method for admin route: " + method)

    def admin_route_inner(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            global n_valid_admin_requests
            params = get_form_data() if method == "post" else get_args_data()
            if (
                params.get("auth_key")
                == hashlib.sha512(
                    (str(n_valid_admin_requests) + config["api_key"]).encode()
                ).hexdigest()
            ):
                n_valid_admin_requests += 1
                return func(*args, **kwargs)
            else:
                return jsonify_error("invalid auth key"), 401

        return wrapper

    return admin_route_inner


class User(UserMixin):
    pass


@login_manager.user_loader
def load_user(user_id):
    if user_id in users:
        return users[user_id]
    user = User()
    user.id = user_id
    return users.setdefault(user_id, user)


# flask static file serving was somehow taking forever to respond with the files (on the order of minutes).
# I tried figuring out why, but ended up just hand-rolling my own static file serving because that's quicker.
@app.route("/static/<filename>")
def hacky_request_file_from_static(filename: str):
    if filename.endswith(".css"):
        return send_from_directory("static", filename, mimetype="text/css")
    elif filename.endswith(".js"):
        return send_from_directory(
            "static", filename, mimetype="application/javascript"
        )
    elif filename.endswith(".json"):
        return send_from_directory("static", filename, mimetype="application/json")
    elif filename.endswith(".html"):
        return send_from_directory("static", filename, mimetype="text/html")
    else:
        return "Invalid file type to fetch", 400


@app.route("/api/get-n-valid-admin-requests")
def api_route_get_n_admin_reqs():
    return jsonify(n_valid_admin_requests=n_valid_admin_requests), 200


@app.route("/login")
def login():
    state = secrets.token_urlsafe(16)
    states[state] = time.time()
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
        return redirect(redirect_url)
    return "Error in authentication process", 500


@app.route("/manual-login", methods=["GET", "POST"])
def manual_login():
    if request.method == "GET":
        return render_template("manual_login.html", url_for=url_for), 200
    form = get_form_data()
    login_token = form.get("token")
    if login_token:
        user_id = get_user_id_from_login_token(login_token)
        if isinstance(user_id, dict) and "error" in user_id:
            return jsonify_error(user_id["error"]), 401
        if not user_id:
            return jsonify_error("invalid login token"), 401
        user = User()
        user.id = user_id
        users[user_id] = user
        login_user(user)
        get_readable_name_of_user(
            user_id
        )  # populate cache on login so I can find a user's user id by username
        return redirect(url_for("index"))

    return jsonify_error("missing login token"), 400


@app.route("/logout")
@login_required
def logout():
    users.pop(current_user.id)
    logout_user()
    return redirect(url_for("logged_out"))


@app.route("/logged-out")
def logged_out():
    return render_template("logged_out.html", url_for=url_for), 200


@app.route("/oauth/callback")
def oauth_callback():
    user_id = request.args.get("userId")
    state = request.args.get("state")
    if state not in states:
        return (
            "Invalid login state: Please retry by going back to the login or starting page!",
            401,
        )
    if user_id:
        user = User()
        user.id = user_id
        users[user_id] = user
        login_user(user)
        get_readable_name_of_user(
            user_id
        )  # populate cache on login so I can find a user's user id by username
        return redirect(url_for("index"))
    return "Error in OAuth callback", 401


@app.route("/api/reset-market", methods=["POST"])
@admin_route("POST")
def api_route_reset_market():
    global seed, next_market_step
    with sim_data_lock:
        seed = get_form_data().get("seed", seed)
        next_market_step = 0
        sim_data.clear()
        if isinstance(sim, market.MarketSimMix):
            for s in sim.sims:
                s.market_boost = 1
                s.event_boost = 0.0
                s.price = 0
                s.step_counter = 1
                s.noise_values = {
                    freq: [
                        market.random.gauss(0, s.stddev),
                        market.random.gauss(0, s.stddev),
                    ]
                    for freq in s.noise_frequencies
                }
        else:
            sim.market_boost = 1
            sim.event_boost = 0.0
            sim.price = 0
            sim.step_counter = 1
            sim.noise_values = {
                freq: [
                    market.random.gauss(0, sim.stddev),
                    market.random.gauss(0, sim.stddev),
                ]
                for freq in sim.noise_frequencies
            }
    return jsonify(), 200


@app.route("/api/inject-event", methods=["POST"])
@admin_route("POST")
def api_inject_event():
    form = get_form_data()
    msg = form.get("message")
    impact = form.get("impact")
    sentiment = form.get("sentiment")
    if not isinstance(impact, (int, float)):
        return jsonify_error("impact must be int or float, but is not"), 400
    if not isinstance(sentiment, str):
        return jsonify_error("sentiment must be a string"), 400
    if not isinstance(msg, str):
        return jsonify_error("message must be a string"), 400
    if sentiment not in ("positive", "negative"):
        return jsonify_error(
            "sentiment must be 'positive' or 'negative', not '" + sentiment + "'"
        )
    user_buy_sell_event_queue.append((impact * USER_ACTION_EVENT_SCALE, sentiment, msg))
    return jsonify(), 200


@app.route("/api/set-fake-points", methods=["POST"])
@admin_route("POST")
def api_route_set_fake_points():
    form = get_form_data()
    amount = form.get("amount")
    user_id = form.get("user_id")
    if not isinstance(amount, int):
        return jsonify_error("invalid amount parameter"), 400
    if user_id not in user_id_to_readable_name:
        return (
            jsonify_error(
                "user id does not exist or has not logged in on this platform yet"
            ),
            400,
        )
    user_info = user_management.get_user_info(user_id)
    cc_utils.fake_points[user_info.public_key] = amount
    return jsonify(), 200


@app.route("/api/get-all-users")
@admin_route("GET")
def api_get_all_users():
    out = []
    for user_id in user_id_to_readable_name:
        user_info = user_management.get_user_info(user_id)
        out.append(
            {
                "user_id": user_id,
                "coins_available": cc_utils.get_available_coins(user_info.public_key),
                "fake_points": user_get_fake_points(user_info.public_key),
                "total_points": user_get_available_points(user_info),
                "name": user_id_to_readable_name[user_id],
            }
        )
    return jsonify(out), 200


@app.route("/api/get-user-by-id")
@admin_route("GET")
def api_get_user_by_id():
    user_id = get_args_data().get("user_id")
    if not user_id:
        return jsonify_error("missing user id"), 400
    user_info = user_management.get_user_info(user_id)
    return (
        jsonify(
            user_id=user_id,
            coins_available=cc_utils.get_available_coins(user_info.public_key),
            fake_points=user_get_fake_points(user_info.public_key),
            total_points=user_get_available_points(user_info),
            name=user_id_to_readable_name[user_id],
        ),
        200,
    )


@app.route("/api/get-user-by-name")
@admin_route("GET")
def api_get_user_by_name():
    name = get_args_data().get("name")
    if not name:
        return jsonify_error("missing name"), 400
    for user_id, assoc_name in user_id_to_readable_name.items():
        if name == assoc_name:
            break
    else:
        return jsonify_error("username does not exist"), 400
    user_info = user_management.get_user_info(user_id)
    return (
        jsonify(
            user_id=user_id,
            coins_available=cc_utils.get_available_coins(user_info.public_key),
            fake_points=user_get_fake_points(user_info.public_key),
            total_points=user_get_available_points(user_info),
            name=user_id_to_readable_name[user_id],
        ),
        200,
    )


@app.route("/api/get-scores")
@login_required
def api_route_get_scores():
    user_id = current_user.id
    user_info = user_management.get_user_info(user_id)
    return jsonify(
        coins_available=cc_utils.get_available_coins(user_info.public_key),
        fake_points=user_get_fake_points(user_info.public_key),
        total_points=user_get_available_points(user_info),
    )


@app.route("/api/buy", methods=["POST"])
@login_required
def api_route_buy():
    user_id = current_user.id
    n_coins_bought = get_form_data().get("amount_coins_bought")
    if not isinstance(n_coins_bought, int):
        return jsonify_error("Invalid amount, it must be an integer, but is not"), 400
    n_coins_bought = int(n_coins_bought)
    amount = n_coins_bought * get_coins_to_points_exchange_rate()
    user_info = user_management.get_user_info(user_id)
    user_amount_available = user_get_available_points(user_info)
    if user_amount_available is None:
        return jsonify_error("Unable to retrieve your point score"), 400
    if amount > user_amount_available:
        amount = user_amount_available
        n_coins_bought = int(amount * get_points_to_coins_exchange_rate())
    if n_coins_bought > 0:
        try:
            cc_utils.send_coins(
                beff_jezos, user_info.public_key, n_coins_bought
            )
        except Exception as e:
            return (
                jsonify_error("Exception occured while processing request: " + str(e)),
                400,
            )
        withdraw_points(
            user_info, math.ceil(n_coins_bought * get_coins_to_points_exchange_rate())
        )
        user_buy_sell_event_queue.append(
            (
                n_coins_bought * USER_ACTION_EVENT_SCALE,
                "positive",
                f"user {get_readable_name_of_user(user_id)} bought {n_coins_bought} CC",
            )
        )
        return (
            jsonify(
                coins_available=cc_utils.get_available_coins(user_info.public_key),
                n_coins_bought=n_coins_bought,
                fake_points=user_get_fake_points(user_info.public_key),
            ),
            200,
        )
    return jsonify_error("Too few points"), 400


@app.route("/api/sell", methods=["POST"])
@login_required
def api_route_sell():
    user_id = current_user.id
    amount_coins = get_form_data().get("amount_coins_sold")
    if not isinstance(amount_coins, int):
        return jsonify_error("Invalid amount, must be int, but is not"), 400
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
                user_info, beff_jezos.public_key, amount_coins
            )
        except Exception as e:
            return (
                jsonify_error("Exception occured while processing request: " + str(e)),
                400,
            )
        deposit_points(user_info, n_points_to_recv)
        user_buy_sell_event_queue.append(
            (
                amount_coins * USER_ACTION_EVENT_SCALE,
                "negative",
                f"user {get_readable_name_of_user(user_id)} sold {amount_coins} CC",
            )
        )
        return (
            jsonify_update(
                coins_available=cc_utils.get_available_coins(user_info.public_key),
                n_coins_sold=amount_coins,
                fake_points=user_get_fake_points(user_info.public_key),
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
            (
                jsonify(
                    market_steps=sim_data_to_json(sim_data[n - next_market_step :]),
                    n_retrieved=next_market_step - n,
                    next_market_step=next_market_step,
                )
                if not current_user.is_authenticated
                or not user_broker_notifications.get(current_user.id)
                else jsonify(
                    market_steps=sim_data_to_json(sim_data[n - next_market_step :]),
                    n_retrieved=next_market_step - n,
                    next_market_step=next_market_step,
                    broker_notification=user_broker_notifications[current_user.id].pop(
                        0
                    ),
                )
            ),
            200,
        )


@app.route("/api/market/next-market-step")
def api_route_get_next_market_step():
    return jsonify(next_market_step=next_market_step), 200


# for correctness, I'd have to add or withdraw cc to a special broker AG pub key...
# except that Beff Jezos owns the CamlCoin Broker AG!!!!!
@app.route("/api/broker/open-position", methods=["POST"])
@login_required
def api_route_broker_open_position():
    user_id = current_user.id
    if user_id in broker_positions:
        return jsonify_error("User already has open position"), 400

    user_info = user_management.get_user_info(user_id)

    leverage = get_form_data().get("leverage")
    if not isinstance(leverage, int):
        return jsonify_error("Invalid leverage, must be int, but is not"), 400

    leverage = int(leverage)

    a, b = get_min_backup_capital_for_leverage(leverage)
    min_required_backup_capital = a * get_coins_to_points_exchange_rate() + b
    if (
        user_get_fake_points(user_info.public_key) < min_required_backup_capital
        and user_get_available_points(user_info) < min_required_backup_capital
    ):
        return (
            jsonify_error("User has too little base capital to back up his position"),
            400,
        )

    real_points = user_get_available_points(user_info)
    leverage_is_allowed, min_points_req_for_leverage = (
        requested_leverage_allowed_with_points(real_points, leverage)
    )
    if not leverage_is_allowed:
        return (
            jsonify_error(
                f"For this amount of leverage, you need at least {min_points_req_for_leverage} points"
            ),
            400,
        )

    broker_positions[user_id] = (
        leverage,
        get_coins_to_points_exchange_rate(),
        time.time(),
    )
    user_buy_sell_event_queue.append(
        (
            leverage * USER_ACTION_EVENT_SCALE,
            "positive",
            f"CamlCoin Broker AG bought {leverage} CC for user {get_readable_name_of_user(user_id)}",
        )
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
    leverage, *_ = broker_positions[user_id]
    net_earnings = user_close_position(user_info)
    user_buy_sell_event_queue.append(
        (
            leverage * USER_ACTION_EVENT_SCALE,
            "negative",
            f"CamlCoin Broker AG sold {leverage} CC for user {get_readable_name_of_user(user_id)}",
        )
    )
    return (
        jsonify(
            net_earnings=net_earnings,
            coins_available=cc_utils.get_available_coins(user_info),
        ),
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
        return jsonify_error("no open position"), 400


@app.route("/api/broker/get-params")
def api_route_get_broker_params():
    return jsonify(
        open_fee=BROKER_OPEN_FEE,
        keep_fee=BROKER_KEEP_FEE,
        keep_fee_interval=BROKER_KEEP_FEE_INTERVAL_SECS,
        max_hold_time=BROKER_MAX_HOLD_TIME_SECS,
    )


@app.route("/api/get-leaderboard")
def api_get_leaderboard():
    user_amounts = [
        {
            "score": cc_utils.get_available_fake_points(
                user_management.get_user_info(user_id).public_key
            ),
            "name": get_readable_name_of_user(user_id),
        }
        for user_id in users
    ]
    return sorted(user_amounts, key=lambda info: info["score"], reverse=True)[
        :LEADERBOARD_SIZE
    ]


@app.route("/leaderboard")
def view_leaderboard():
    return render_template("leaderboard.html"), 200


@app.route("/")
@login_required
def index():
    return render_template("index.html"), 200


states = {}
STATE_TIMEOUT = 15

user_buy_sell_event_queue = []
USER_ACTION_EVENT_SCALE = 0.0001

broker_positions = {}
BROKER_OPEN_FEE = 0.15  # amount of points it costs to open a position at leverage=1
BROKER_KEEP_FEE = (
    0.1  # amount of points it costs to keep an open position at leverage=1
)
BROKER_KEEP_FEE_INTERVAL_SECS = 10  # interval in which BROKER_KEEP_FEE is charged
BROKER_MAX_HOLD_TIME_SECS = 120  # after this amount of time, auto-sell (so users can't forget about their position and loose everything to keep fees)

SIM_BATCHES_PER_SECOND = 2  # num sim batches per second
SIM_BATCH_SIZE = 1  # how many steps are computed in one go to avoid locking too often
seed = 43

sim = market.MarketSimMix(
    (
        market.MarketSim(  # this sim provides high frequency "day trading" dynamics that can be changed by fake X posts etc
            base_value=100.0,
            bounce_back_value=40.0,
            upper_bounce_back_value=400.0,
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
            event_decay_factor=0.99999999,
            event_impact=0.0075,
            event_boost_weight=50,
            event_prob=0.08,
            event_change_trend_min_stds_from_mean=0.4,
            event_change_trend_weight=1.0,
        ),
        market.MarketSim(  # this simulation provides low frequency hourly / 20-minute trends that must not be influenced by events
            base_value=100,
            bounce_back_value=40,
            upper_bounce_back_value=600.0,
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
            event_decay_factor=0.99999999,
            event_impact=0.015,
            event_boost_weight=50,
            event_prob=0.01,
            event_change_trend_min_stds_from_mean=0.2,
            event_change_trend_weight=0.1,
        ),
    ),
    (user_buy_sell_event_queue, user_buy_sell_event_queue),
)
MIN_REACHABLE_PRICE = math.ceil(sum(s.bounce_back_value for s in sim.sims))

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
# user_management.create_users_table()

MAX_PROFIT_POINTS = None
if "max_profit_points" in config:
    MAX_PROFIT_POINTS = config["max_profit_points"]

broker_leverage_min_backup_capital_pairs = (
    [  # inner dict is (a, b) pair where min backup capital is a * leverage + b
        (
            5000,
            (100, MAX_PROFIT_POINTS) if MAX_PROFIT_POINTS is None else (0, MAX_PROFIT_POINTS + 1000),
        ),
        (
            3000,
            (50, MIN_REACHABLE_PRICE),
        ),
        (
            1000,
            (20, MIN_REACHABLE_PRICE),
        ),
        (
            300,
            (5, MIN_REACHABLE_PRICE),
        ),
        (
            200,
            (4, MIN_REACHABLE_PRICE),
        ),
        (
            100,
            (3, MIN_REACHABLE_PRICE),
        ),
        (
            40,
            (2, MIN_REACHABLE_PRICE),
        ),
        (
            10,
            (1, MIN_REACHABLE_PRICE),
        ),
        (0, (0, MIN_REACHABLE_PRICE)),
    ]
)

user_id_to_readable_name = {}
user_broker_notifications = {}
sim_data = []
next_market_step = 0
SIM_DATA_BACKLOG_SIZE = 50_000
users = {}
sim_data_lock = threading.Lock()

n_valid_admin_requests = 0

LEADERBOARD_SIZE = 10

sim_thread = threading.Thread(target=restart_on_crash(run_sim))
bg_service_thread = threading.Thread(target=restart_on_crash(run_background_services))

sim_thread.start()
bg_service_thread.start()

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8000, debug=False, use_reloader=False, threaded=True)
