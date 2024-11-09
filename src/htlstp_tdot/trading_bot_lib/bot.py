import requests
import threading
import time
from copy import deepcopy


class TradingBot:
    def __init__(self, domain, user_token, after_request_sleep_time=0.25):
        self.domain = domain
        self.user_token = user_token
        self.new_data_callbacks = []
        self.error_callbacks = []
        self.data_history = []
        self.history_size_constraints = []
        self.max_history_size = 500
        self.next_market_step = 0
        self.is_running = False
        self.after_request_sleep_time = after_request_sleep_time
        self.session = requests.Session()  # Use a session to persist login state

    def on_new_data(self, *fn, history_size=500):
        if fn:
            func = fn[0]
            self.new_data_callbacks.append((func, history_size))
            self.history_size_constraints.append(history_size)
            self.max_history_size = max(self.history_size_constraints)
            return func
        else:
            def decorator(func):
                self.new_data_callbacks.append((func, history_size))
                self.history_size_constraints.append(history_size)
                self.max_history_size = max(self.history_size_constraints)
                return func
            return decorator

    def on_error(self, func):
        self.error_callbacks.append(func)
        return func

    def _login(self):
        url = f"{self.domain}/manual-login"
        response = self.session.post(url, json={"token": self.user_token})
        response.raise_for_status()

    def _fetch_data(self):
        url = f"{self.domain}/api/market/steps-since/{self.next_market_step}"
        response = self.session.get(url)
        response.raise_for_status()
        return response.json()

    def _update_data(self, data_points, next_market_step):
        self.data_history.extend(data_points)
        while len(self.data_history) > self.max_history_size:
            self.data_history.pop(0)
        self.next_market_step = next_market_step

    def _notify_callbacks(self):
        data_copy = deepcopy(self.data_history)
        for callback, history_size in self.new_data_callbacks:
            callback(data_copy[-history_size:])

    def _get_next_market_step(self):
        url = f"{self.domain}/api/market/next-market-step"
        response = self.session.get(url)
        response.raise_for_status()
        return max(response.json()["next_market_step"] - self.max_history_size, 0)

    def start(self, run_on_thread=False):
        self._login()
        self.is_running = True
        if run_on_thread:
            threading.Thread(target=self._run, daemon=True).start()
        else:
            self._run()

    def buy(self, amount: int):
        url = f"{self.domain}/api/buy"
        response = self.session.post(url, json={"amount_coins_bought": amount})
        response.raise_for_status()
        return response.json()["n_coins_bought"]

    def sell(self, amount: int):
        url = f"{self.domain}/api/sell"
        response = self.session.post(url, json={"amount_coins_sold": amount})
        response.raise_for_status()
        return response.json()["n_coins_sold"]

    def current_exchange_rate(self):
        if not self.data_history:
            url = f"{self.domain}/api/market/current-exchange-rate"
            response = self.session.get(url)
            response.raise_for_status()
            return response.json()["exchange_rate"]

    def _run(self):
        self.next_market_step = self._get_next_market_step()
        while self.is_running:
            try:
                data = self._fetch_data()
                if "error" not in data:
                    market_steps = data.get('market_steps', [])
                    next_market_step = data.get('next_market_step', self.next_market_step)
                    if market_steps:
                        self._update_data(market_steps, next_market_step)
                        self._notify_callbacks()
                time.sleep(self.after_request_sleep_time)
            except Exception as e:
                for ec in self.error_callbacks:
                    ec(e)


if __name__ == "__main__":
    bot = TradingBot(domain="http://127.0.0.1:8000", user_token=input("login token: "))
    has_bought, has_sold = False, False

    @bot.on_new_data()
    def handle_data(data):
        global has_bought, has_sold
        assert len(data) == 500
        print("Received new data points: len =", len(data))
        if not has_bought:
            n_bought = bot.buy(1)
            if n_bought:
                has_bought = True
                print("bought!")
            else:
                print("buy failed")
        elif not has_sold:
            n_sold = bot.sell(1)
            if n_sold:
                has_sold = True
                print("sold!")
            else:
                print("sell failed")

    @bot.on_new_data
    def handle_data2(data):
        print("second handler")

    @bot.on_new_data(history_size=1)
    def handler3(data):
        print("last data point:", data[0])

    @bot.on_error
    def handle_error(e):
        print(e)

    bot.start()
