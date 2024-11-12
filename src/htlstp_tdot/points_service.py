import json
import time
import uuid
import os
import secrets
import string
import hashlib
import threading
import sqlite3
from flask import Flask, request, jsonify, render_template, redirect, url_for, send_from_directory
from urllib.parse import urlencode

app = Flask(__name__)

DB_PATH = "data/points_service_state.db"


def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            user_id TEXT PRIMARY KEY,
            points INTEGER DEFAULT 0,
            email TEXT,
            password_hash TEXT,
            salt TEXT,
            name TEXT,
            created_at REAL
        )
    """
    )
    conn.commit()
    conn.close()


def get_user_id_by_token(token):
    return token_to_user_id.get(token)


def create_user_id():
    return str(uuid.uuid4())


def deposit_points(user_id, points):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    # Handle infinite points user
    if user_id == "00000000-0000-0000-0000-000000000000":
        c.execute("SELECT points FROM users WHERE user_id = ?", (user_id,))
        infinite_points = c.fetchone()[0]
        conn.close()
        return {"points": infinite_points}

    c.execute(
        "UPDATE users SET points = points + ? WHERE user_id = ?",
        (points, user_id),
    )
    conn.commit()

    # Get updated balance
    c.execute("SELECT points FROM users WHERE user_id = ?", (user_id,))
    new_balance = c.fetchone()[0]
    conn.close()
    return {"points": new_balance}


def withdraw_points(user_id, points):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    # Handle infinite points user
    if user_id == "00000000-0000-0000-0000-000000000000":
        c.execute("SELECT points FROM users WHERE user_id = ?", (user_id,))
        infinite_points = c.fetchone()[0]
        conn.close()
        return {"points": infinite_points}

    # Check current balance and withdraw
    c.execute("SELECT points FROM users WHERE user_id = ?", (user_id,))
    current_balance = c.fetchone()[0]
    if current_balance < points:
        conn.close()
        return {"error": "Insufficient points"}

    c.execute(
        "UPDATE users SET points = points - ? WHERE user_id = ?", (points, user_id)
    )
    conn.commit()

    # Get updated balance
    c.execute("SELECT points FROM users WHERE user_id = ?", (user_id,))
    new_balance = c.fetchone()[0]
    conn.close()
    return {"points": new_balance}


def get_user_points(user_id):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT points FROM users WHERE user_id = ?", (user_id,))
    result = c.fetchone()
    conn.close()
    return {"points": result[0]} if result else {"error": "User not found"}


def get_user_email(user_id):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT email FROM users WHERE user_id = ?", (user_id,))
    result = c.fetchone()
    conn.close()
    return {"value": result[0]} if result else {"error": "User not found"}


def get_user_name(user_id):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT name FROM users WHERE user_id = ?", (user_id,))
    result = c.fetchone()
    conn.close()
    return {"value": result[0]} if result else {"error": "User not found"}


def generate_salt_and_password_hash(password):
    """Generate a salt and hash the password with it."""
    salt = os.urandom(16).hex()
    password_hash = hashlib.sha256((salt + password).encode("utf-8")).hexdigest()
    return password_hash, salt


def get_salt_and_password_hash(user_id):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT salt, password_hash FROM users WHERE user_id = ?", (user_id,))
    result = c.fetchone()
    conn.close()
    return (
        {"salt": result[0], "password_hash": result[1]}
        if result
        else {"error": "User not found"}
    )


def create_user(email, username, password):
    """Store the password hash and salt in the database."""
    if not email:
        return jsonify(error="email must not be empty")
    elif not password:
        return jsonify(error="password must not be empty")
    elif not username:
        return jsonify(error="username must not be empty")
    elif "user_id" in get_user_id_from_email(email):
        return jsonify(error="Email already used on a different account")

    password_hash, salt = generate_salt_and_password_hash(password)
    conn = sqlite3.connect("data/points_service_state.db")
    c = conn.cursor()
    c.execute(
        "INSERT INTO users (user_id, points, email, password_hash, salt, name, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (create_user_id(), 0, email, password_hash, salt, username, time.time()),
    )
    conn.commit()
    conn.close()
    return {}


def get_user_id_from_email(email):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT user_id FROM users WHERE email = ?", (email,))
    result = c.fetchone()
    conn.close()
    return {"user_id": result[0]} if result else {"error": "User not found"}


def get_api_key():
    auth_header = request.headers.get("Authorization", "")
    return auth_header.split(" ")[-1] if "Bearer" in auth_header else None


def user_check_password(user_id, password):
    out = get_salt_and_password_hash(user_id)
    if "error" in out:
        return jsonify(error=out["error"]), 400
    salt, password_hash = out["salt"], out["password_hash"]
    return (
        hashlib.sha256((salt + password).encode("utf-8")).hexdigest() == password_hash
    )


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


@app.route("/api/public/user/deposit", methods=["POST"])
def api_deposit_points():
    api_key = get_api_key()
    if api_key != config["api_key"]:
        return jsonify({"error": "Unauthorized"}), 401

    data = request.json
    user_id = data.get("userId")
    points = data.get("points")
    out = deposit_points(user_id, points)
    status_code = 400 if "error" in out else 200
    return jsonify(out), status_code


@app.route("/api/public/user/withdraw", methods=["POST"])
def api_withdraw_points():
    api_key = get_api_key()
    if api_key != config["api_key"]:
        return jsonify({"error": "Unauthorized"}), 401

    data = request.json
    user_id = data.get("userId")
    points = data.get("points")
    out = withdraw_points(user_id, points)
    status_code = 400 if "error" in out else 200
    return jsonify(out), status_code


@app.route("/api/public/user/points/<user_id>", methods=["GET"])
def api_get_points(user_id):
    api_key = get_api_key()
    if api_key != config["api_key"]:
        return jsonify({"error": "Unauthorized"}), 401
    out = get_user_points(user_id)
    status_code = 400 if "error" in out else 200
    return jsonify(out), status_code


@app.route("/api/auth/user", methods=["GET"])
def api_get_user_id():
    token = request.args.get("token")
    user_id = get_user_id_by_token(token)
    if user_id:
        return jsonify({"value": user_id})
    return jsonify({"error": "Invalid or expired token"}), 404


@app.route("/api/public/user/email/<user_id>", methods=["GET"])
def api_get_user_email(user_id):
    api_key = get_api_key()
    if api_key != config["api_key"]:
        return jsonify({"error": "Unauthorized"}), 401
    out = get_user_email(user_id)
    status_code = 400 if "error" in out else 200
    return jsonify(out), status_code


@app.route("/api/public/user/name/<user_id>", methods=["GET"])
def api_get_user_name(user_id):
    api_key = get_api_key()
    if api_key != config["api_key"]:
        return jsonify({"error": "Unauthorized"}), 401
    out = get_user_name(user_id)
    status_code = 400 if "error" in out else 200
    return jsonify(out), status_code


@app.route("/api/auth/createRedirectUri", methods=["GET"])
def create_redirect_uri():
    api_key = get_api_key()
    if api_key != config["api_key"]:
        return jsonify(error="Unauthorized"), 401
    state = request.args.get("state")
    redirect_uri = request.args.get("redirect_uri")
    if not redirect_uri or not state:
        return jsonify(error="Bad request"), 400
    return jsonify(value=url_for("login", state=state, redirect_uri=redirect_uri, _external=True))


@app.route("/auth/login", methods=["GET", "POST"])
def login():
    if request.method == "GET":
        return render_template("points_service_login.html")
    state = request.args.get("state")
    callback_uri = request.args.get("redirect_uri")
    email = request.form.get("email")
    password = request.form.get("password")
    out = get_user_id_from_email(email)
    if "error" in out:
        return redirect(url_for("sign_up", redirect_uri=callback_uri, state=state))
    user_id = out["user_id"]

    if user_check_password(user_id, password):
        if callback_uri and state:
            return redirect(f"{callback_uri}?{urlencode({'state': state, 'userId': user_id})}")
        return render_template("points_service_logged_in.html", points=get_user_points(user_id).get("points"))
    return jsonify(error="Incorrect password"), 401


@app.route("/auth/signup", methods=["GET", "POST"])
def sign_up():
    if request.method == "GET":
        return render_template("points_service_signup.html")
    callback_uri = request.args.get("redirect_uri")
    state = request.args.get("state")
    email = request.form.get("email")
    username = request.form.get("username")
    password = request.form.get("password")
    out = create_user(email, username, password)
    if "error" in out:
        return jsonify(error=out["error"]), 400
    out = get_user_id_from_email(email)
    if "error" in out:
        return jsonify(error=out["error"]), 400
    user_id = out["user_id"]
    if user_check_password(user_id, password):
        if callback_uri and state:
            return redirect(f"{callback_uri}?{urlencode({'state': state, 'userId': user_id})}")
        return render_template("points_service_signup_successful.html")
    return jsonify(error="Incorrect password, sign-up messed something up; sry :("), 401


@app.route("/auth/createLoginToken", methods=["GET", "POST"])
def get_login_token():
    if request.method == "GET":
        return render_template("points_service_get_token.html")
    email = request.form.get("email")
    password = request.form.get("password")
    out = get_user_id_from_email(email)
    if "error" in out:
        return jsonify(error=out["error"]), 400
    user_id = out["user_id"]
    if user_check_password(user_id, password):
        chars = string.digits + string.ascii_lowercase
        token = ''.join(secrets.choice(chars) for _ in range(TOKEN_SIZE))
        token_to_user_id[token] = user_id
        return render_template("points_service_show_token.html", token=token)
    return jsonify(error="Incorrect password"), 401


@app.route("/", methods=["GET"])
def index():
    return render_template("points_service_index.html")


def run_background_services():
    while True:
        t = time.time()
        for token in set(token_to_user_id.keys()):
            if token_create_time[token] - t > TOKEN_LIFETIME:
                token_to_user_id.pop(token)
                token_create_time.pop(token)
        time.sleep(BACKGROUND_SERVICE_INTERVAL)


TOKEN_LIFETIME = 120
BACKGROUND_SERVICE_INTERVAL = 60
TOKEN_SIZE = 6
with open("config.json") as f:
    config = json.load(f)
init_db()
token_to_user_id = {}
token_create_time = {}
data_dump_thread = threading.Thread(target=run_background_services)
data_dump_thread.start()


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8001, debug=False, use_reloader=False, threaded=True)
