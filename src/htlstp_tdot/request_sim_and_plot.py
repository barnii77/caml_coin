import requests
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

vs = []
window_size = 1000  # Number of points to display in the window
next_market_step = 0

# Prepare the figure and axis
fig, ax = plt.subplots()
line, = ax.plot([])  # Empty plot initially
ax.set_xlim(0, window_size)  # Fixed window size of 1000 points


def init():
    """Initialize the background of the plot."""
    line.set_data([], [])
    return line,


def update(frame):
    """Update the plot with the next price."""
    global next_market_step
    try:
        resp = requests.get("http://127.0.0.1:5000/api/market/steps-since/" + str(next_market_step))
    except Exception as e:
        print(str(e))
        return line,
    out = resp.json()
    print(out)
    if out.get("error"):
        return line,
    next_market_step = out["next_market_step"]
    for i in out["market_steps"]:
        price = i["exchange_rate"]
        vs.append(price)

    current_data = vs
    if len(vs) > window_size:
        current_data = vs[-window_size:]

    # Update the x and y data for the plot
    line.set_data(range(len(current_data)), current_data)
    ax.set_ylim(0, max(current_data) + 0.1)
    return line,


# Create the animation
ani = FuncAnimation(
    fig, update, frames=range(1000), init_func=init, blit=False, interval=50
)

# ani.save("market_simulation.mp4", writer="ffmpeg", fps=30)
plt.show()
