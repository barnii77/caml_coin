import secrets
import json
from crypt import methods

import requests
from flask import Flask, url_for, redirect, request, render_template
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


def get_user_info(user_id):
    # TODO make db to store the users secrets and retrieve the users cached num caml coins, have thread running in background reloading the actual
    # TODO balances from the blockchain
    return ()


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
        return redirect(redirect_url)
    return "Error in authentication process", 500


@app.route("/oauth_callback")
def oauth_callback():
    state = request.args.get("state")
    user_id = request.args.get("userId")
    if state and user_id:
        user = User(user_id, state)
        login_user(user)
        return redirect(url_for("index"))
    return "Error in OAuth callback", 400


@app.route("/buy", methods=["GET", "POST"])
@login_required
def buy():
    if request.method == "GET":
        return render_template("buy.html"), 200
    user_id = current_user.id
    amount = request.args.get("amount")
    user_amount_available = user_get_available_points(user_id)
    if not user_amount_available:
        return "Unable to retrieve your point score", 400
    if user_amount_available >= amount:
        requests.post(
            f"{config['points_service']}/api/public/user/withdraw",
            json.dumps({"userId": user_id, "points": amount}),
            headers=api_headers,
        )
    return redirect(url_for("index")), 200


@app.route("/sell", methods=["GET", "POST"])
def sell():
    if request.method == "GET":
        return render_template("sell.html"), 200
    user_id = current_user.id
    amount = request.args.get("amount")
    user_amount_available = user_get_available_points(user_id)
    if not user_amount_available:
        return "Unable to retrieve your point score", 400
    if user_amount_available >= amount:
        requests.post(
            f"{config['points_service']}/api/public/user/deposit",
            json.dumps({"userId": user_id, "points": amount}),
            headers=api_headers,
        )
    return redirect(url_for("index")), 200


@app.route("/index")
def index():
    user_id = current_user.id
    info = get_user_info(user_id)
    return render_template("index.html"), 200


if __name__ == "__main__":
    app.run(debug=True)
