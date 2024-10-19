import random
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from typing import Union, Optional

# Updated lists of names with more prominent tech figures and investors
first_names = [
    "Satoshi",
    "Vitalik",
    "Charlie",
    "Elon",
    "Nakamoto",
    "Andreas",
    "Hal",
    "Craig",
    "Changpeng",
    "Tyler",
    "Cameron",
    "Barry",
    "Jihan",
    "Jed",
    "Nick",
    "Zooko",
    "Gavin",
    "Roger",
    "Elun",
    "Satashi",
    "Vitaliq",
    "Charly",
    "Elin",
    "Andrzej",
    "Crayg",
    "Changpin",
    "Tiler",
    "Caramon",
    "Bary",
    "Jihen",
    "Jedidiah",
    "Nik",
    "Zucko",
    "Rojer",
    "Elorn",
    "Mark",
    "Sam",
    "Tim",
    "Bill",
    "Jeff",
    "Larry",
    "Sergey",
    "Sundar",
    "Mark",
    "Elorn",
    "Reid",
    "Peter",
    "Marc",
    "Jack",
    "Sheryl",
]

last_names = [
    "Nakamura",
    "Buterin",
    "Hoskinson",
    "Musk",
    "Antonopoulos",
    "Finney",
    "Wright",
    "Zhao",
    "Lee",
    "Winklevoss",
    "Silbert",
    "Wu",
    "McCaleb",
    "Szabo",
    "Wilcox",
    "Wood",
    "Ver",
    "Musko",
    "Writ",
    "Zhou",
    "Li",
    "Winkelvoss",
    "Silber",
    "Wou",
    "Mcaleb",
    "Sabbo",
    "Wilcock",
    "Wod",
    "Vear",
    "Mosk",
    "Wrighton",
    "Butterin",
    "Husk",
    "Zuckerberg",
    "Altman",
    "Cuban",
    "Cook",
    "Gates",
    "Bezos",
    "Page",
    "Brin",
    "Pichai",
    "Benioff",
    "Cuban",
    "Hoffman",
    "Thiel",
    "Andreessen",
    "Dorsey",
    "Sandberg",
]

# Platforms for posting
platforms = [
    "Twitter",
    "Reddit",
    "LinkedIn",
    "Medium",
    "Facebook",
    "Instagram",
    "YouTube",
    "TikTok",
    "Telegram",
    "Threads",
]

# Extended positive sentences
positive_sentences = [
    "This coin is like a rocket ship to the moon, you better hop on!",
    "The community is so strong, they could crowdfund a moon colony.",
    "It’s like digital gold, only shinier and smarter.",
    "There’s no stopping this one, it's like the internet in the '90s.",
    "It’s so decentralized, even your grandma could run a node.",
    "The roadmap is longer than a Tolstoy novel, and twice as impressive.",
    "You're not just buying a coin, you're buying into the future.",
    "It's the kind of innovation that makes Satoshi smile from wherever he is.",
    "If this isn’t the next big thing, nothing is.",
    "You know it’s good when even your taxi driver is asking how to buy some.",
    "It’s so green, it makes Tesla look like a coal mine.",
    "With this coin, even your pet could retire early.",
    "Hodl this one, and you’ll need a second wallet for all the gains.",
    "This coin makes DeFi look like child's play.",
    "It's the most secure chain out there – hackers get scared just looking at it.",
    "Even whales are jealous of how big this coin is getting.",
    "It’s like a faucet of passive income – but with jets.",
    "If it were any more decentralized, it would dissolve into thin air.",
    "This is the Lamborghini of cryptocurrencies.",
    "You can feel the gains just by holding it in your wallet.",
]

# Extended negative sentences
negative_sentences = [
    "This coin’s future is as bright as a broken lightbulb.",
    "It’s about as decentralized as a pizza shop.",
    "It’s tanking faster than a lead balloon.",
    "If you’re looking for a quick way to lose money, this is it.",
    "The roadmap looks like a treasure map… but without the treasure.",
    "This is the kind of coin that even rug pulls don't bother with.",
    "It’s so volatile, it gives rollercoasters a run for their money.",
    "Investing in this coin is like throwing money into a black hole.",
    "It’s a ghost town – even the developers have ghosted.",
    "The only thing this coin is mining is disappointment.",
    "It's dropping faster than a stone tied to an anchor.",
    "You'd be better off investing in rocks – at least they don't move.",
    "It’s been outperformed by actual memes.",
    "Even your toaster has more utility than this coin.",
    "It’s not just going to zero, it’s digging a hole past it.",
    "This coin's security is so bad, it’s practically an invitation to hackers.",
    "Buying this coin is like trying to catch a falling knife.",
    "The only roadmap here is the one to bankruptcy.",
    "If vaporware had a mascot, it’d be this coin.",
]


# Generate a random name
def generate_name():
    return f"{random.choice(first_names)} {random.choice(last_names)}"


# Generate sentences based on sentiment
def generate_sentences(sentiment):
    if sentiment == "positive":
        return random.sample(positive_sentences, random.randint(1, 3))
    else:
        return random.sample(negative_sentences, random.randint(1, 3))


# Generate a random platform
def generate_platform():
    return random.choice(platforms)


# Generate a compliment or insult based on randomly chosen sentiment
def generate_cryptocurrency_opinion():
    name = generate_name()
    platform = generate_platform()
    sentiment = random.choice(["positive", "negative"])
    sentences = generate_sentences(sentiment)
    return sentiment, f"{name} posted on {platform}: " + " ".join(sentences)


def get_market_event():
    event_generators = [generate_cryptocurrency_opinion]
    return random.choice(event_generators)()


def _interpolate(start, end, t):
    """Linear interpolation between two points."""
    return start + (end - start) * t


class MarketSim:
    def __init__(
        self,
        decay_factor=0.5,
        market_boost_increase_factor=1.0,
        stddev=1.0,
        seed=None,
        freqs: Optional[Union[list, int]] = 8,
        event_impact=1.0,
        event_prob=0.05,
    ):
        if freqs is None:
            freqs = 8
        if isinstance(freqs, int):
            freqs = [2**i for i in range(freqs)]
        # Initialize market state
        self.market_boost = 1
        self.market_boost_increase_factor = market_boost_increase_factor
        self.price = 0
        self.event_boost = 0.0
        self.decay_factor = decay_factor
        self.stddev = stddev
        self.seed = seed
        self.event_prob = event_prob
        self.event_impact = event_impact
        if seed is not None:
            random.seed(seed)

        # Store noise values for different frequencies
        self.max_freq = max(freqs)
        self.noise_frequencies = [
            1 / i for i in freqs
        ]  # Frequencies: 1, 1/2, 1/4, ..., 1/max_freq
        self.inverse_frequencies = [i for i in freqs]  # Time steps between samples
        self.noise_values = {
            freq: self._generate_noise(freq) for freq in self.noise_frequencies
        }
        self.step_counter = 0

    def _generate_noise(self, frequency):
        """Generates two initial Gaussian noise values for each frequency."""
        return [random.gauss(0, self.stddev), random.gauss(0, self.stddev)]

    def _get_fractal_noise(self):
        total_noise = 0.0

        # Sum up interpolated noise values from each frequency, weighted by 1/frequency
        for freq, inv_freq in zip(self.noise_frequencies, self.inverse_frequencies):
            # Update noise for next step if we've hit the next sample point
            if self.step_counter % inv_freq == 0:
                self.noise_values[freq][0] = self.noise_values[freq][1]
                self.noise_values[freq][1] = random.gauss(0, self.stddev)

            alpha = (self.step_counter % inv_freq) / inv_freq

            start_noise = self.noise_values[freq][0]
            end_noise = self.noise_values[freq][1]
            interpolated_noise = _interpolate(start_noise, end_noise, alpha)

            total_noise += interpolated_noise * inv_freq

        # Scale noise by market boost
        total_noise *= self.market_boost

        return total_noise

    def step(self):
        # Continue generating noise until the price is above zero
        while True:
            noise = self._get_fractal_noise()

            # Update price with noise and current event boost/damp
            self.price = noise + self.event_boost

            # If the price is above zero, proceed; otherwise, resample the noise
            if self.price > 0:
                self.event_boost *= self.decay_factor
                self.market_boost *= self.market_boost_increase_factor
                break
            else:
                self._resample_all_noise()

        # Handle events
        if random.random() < self.event_prob:
            sentiment, message = get_market_event()
            self._handle_event(sentiment)
        else:
            sentiment, message = None, None

        self.step_counter += 1
        return self.price, (
            (sentiment, message)
            if sentiment is not None and message is not None
            else None
        )

    def _resample_all_noise(self):
        """Resample all noise values for every frequency."""
        for freq in self.noise_frequencies:
            self.noise_values[freq][1] = 0.0001 + abs(
                random.gauss(0, self.stddev / 100)
            )
            self.event_boost = 0
            self.market_boost = 1
        self.step_counter = 0

    def _handle_event(self, sentiment):
        # Generate a random boost/damp value for events
        event_impact = random.uniform(1 * self.event_impact, 5 * self.event_impact)
        if sentiment == "positive":
            self.event_boost += event_impact
        else:
            self.event_boost -= event_impact


# Example usage
# vs = []
# window_size = 1000  # Number of points to display in the window
# sim = MarketSim(
#     market_boost_increase_factor=1.0,
#     seed=42,
#     stddev=0.04,
#     freqs=[1, 2, 4, 8, 32, 64, 128, 256, 512],
#     decay_factor=0.96,
#     event_impact=1.25,
#     event_prob=0.02,
# )
#
# # Prepare the figure and axis
# fig, ax = plt.subplots()
# line, = ax.plot([])  # Empty plot initially
# ax.set_xlim(0, window_size)  # Fixed window size of 1000 points
# ax.set_ylim(0, 5)  # Set y-limits, adjust as needed for your data
#
#
# def init():
#     """Initialize the background of the plot."""
#     line.set_data([], [])
#     return line,
#
#
# def update(frame):
#     """Update the plot with the next price."""
#     # Get the next price from the simulation
#     for i in range(10):
#         price = sim.step()[0]
#         vs.append(price)
#
#         # Show only the last `window_size` points in the plot
#         if len(vs) > window_size:
#             current_data = vs[-window_size:]
#         else:
#             current_data = vs
#
#     # Update the x and y data for the plot
#     line.set_data(range(len(current_data)), current_data)
#     ax.set_ylim(min(current_data) - 0.1, max(current_data) + 0.1)  # Adjust y-limits dynamically
#     return line,
#
#
# # Create the animation
# ani = FuncAnimation(
#     fig, update, frames=range(1000), init_func=init, blit=True, interval=50
# )
#
# # ani.save("market_simulation.mp4", writer="ffmpeg", fps=30)
# plt.show()
