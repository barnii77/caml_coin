import secrets
from flask import Flask
from flask_login import LoginManager, UserMixin

app = Flask(__name__)

app.secret_key = secrets.token_urlsafe(24)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'


class User(UserMixin):
    def __init__(self, id):
        self.id = id


@login_manager.user_loader
def load_user(user_id):
    return User(user_id)
