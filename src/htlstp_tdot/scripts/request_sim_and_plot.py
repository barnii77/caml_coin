import requests
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import FancyBboxPatch

vs = []
window_size = 1000  # Number of points to display in the window
next_market_step = 0
prev_text = ""
prev_sentiment = "positive"

# Try setting an emoji-capable font
emoji_font = "Noto Color Emoji"  # You can change this to other available emoji fonts
fallback_font = "Arial"  # Fallback font in case the emoji font is unavailable
emoji_font = fallback_font

try:
    # Set the default font globally to one that supports emojis
    plt.rcParams['font.family'] = emoji_font
except Exception:
    plt.rcParams['font.family'] = fallback_font
    print(f"Warning: '{emoji_font}' not found. Falling back to '{fallback_font}'.")

try:
    resp = requests.get("http://127.0.0.1:5000/api/market/next-market-step")
except Exception as e:
    print(str(e))
else:
    out = resp.json()
    next_market_step = out["next_market_step"]

# Prepare the figure and axis
fig, ax = plt.subplots(figsize=(10, 6))  # Adjust figure size if needed

# Adjust layout to make space for text
fig.subplots_adjust(top=0.8)

(line,) = ax.plot([])  # Empty plot initially
ax.set_xlim(0, window_size)  # Fixed window size of 1000 points
ax.set_ylim(0, 30)  # Set an initial y-limit for the exchange rate

# Create a separate text area above the graph
text_ax = fig.add_axes([0.1, 0.85, 0.8, 0.1])  # [left, bottom, width, height]
text_ax.set_axis_off()  # Hide the axes for the text area

# Add a background frame for the text, and give it an initial color
frame = FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0.1", transform=text_ax.transAxes, edgecolor="green", facecolor="lightgreen")
text_ax.add_patch(frame)

# Initialize the text box inside the text area
text_box = text_ax.text(
    0.5, 0.5, "", transform=text_ax.transAxes, fontsize=10, ha="center", va="center", wrap=True
)

def init():
    """Initialize the background of the plot."""
    line.set_data([], [])
    return line, frame, text_box

def update(frame_number):
    """Update the plot with the next price and display the JSON."""
    global next_market_step, prev_text, prev_sentiment
    try:
        resp = requests.get(
            "http://127.0.0.1:5000/api/market/steps-since/" + str(next_market_step)
        )
    except Exception as e:
        print(str(e))
        return line, frame, text_box

    out = resp.json()
    print(out)  # Print the incoming JSON for debugging/monitoring
    for market_step in out.get("market_steps", []):
        if market_step.get("event") is not None:
            prev_text = market_step["event"]["message"]
            prev_sentiment = market_step["event"]["sentiment"]

    # Update the text box
    text_box.set_text(prev_text)

    # Always show the frame, but change color based on sentiment
    if prev_sentiment == "positive":
        frame.set_edgecolor("green")
        frame.set_facecolor("lightgreen")
    elif prev_sentiment == "negative":
        frame.set_edgecolor("red")
        frame.set_facecolor("lightcoral")
    else:
        # Neutral color if no sentiment, keep the frame visible
        frame.set_edgecolor("gray")
        frame.set_facecolor("lightgray")

    if out.get("error"):
        return line, frame, text_box

    next_market_step = out["next_market_step"]
    for i in out["market_steps"]:
        price = i["exchange_rate"]
        vs.append(price)

    current_data = vs
    if len(vs) > window_size:
        current_data = vs[-window_size:]

    # Update the x and y data for the plot
    line.set_data(range(len(current_data)), current_data)
    ax.set_ylim(min(current_data), max(current_data) + 0.1)  # Dynamic y-axis adjustment

    return line, frame, text_box

# Create the animation
ani = FuncAnimation(
    fig, update, frames=range(1000), init_func=init, blit=False, interval=50
)

plt.show()
