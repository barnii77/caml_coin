import requests
import json

with open("config.json") as f:
    config = json.load(f)


api_headers = {"Authorization": f"Bearer {config['api_key']}"}
user_id = input("UserID or login token: ")

if len(user_id) == 6:
    resp = requests.get(
        f"{config['points_service']}/api/auth/user?token={user_id}",
        headers=api_headers,
    )
    if not str(resp.status_code).startswith('2'):
        print("when resolving UserID, API responded with code", resp.status_code)
        exit()
    user_id = resp.json().get("value")
    if user_id is None:
        print("API did not respond with UserID")
        exit()
    print("UserID is", user_id)

action = input("action ('deposit' or 'withdraw'): ")
if action not in ("deposit", "withdraw"):
    print("action must be one of ('deposit' or 'withdraw')")
    exit()
points = input("amount of points (positive integer): ")
try:
    points = int(points)
except ValueError:
    print("amount must be an integer")
    exit()

if points <= 0:
    print("amount must be positive")
    exit()

if action == "deposit":
    resp = requests.post(
        f"{config['points_service']}/api/public/user/deposit",
        json={"userId": user_id, "points": points},
        headers=api_headers,
    )
else:
    resp = requests.post(
        f"{config['points_service']}/api/public/user/withdraw",
        json={"userId": user_id, "points": points},
        headers=api_headers,
    )

print("API responded with code", resp.status_code)
