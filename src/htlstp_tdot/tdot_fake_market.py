import random
import math
from typing import Union, Optional, Tuple

first_names = [
    "David",
    "Daniel",
    "Danid",
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
    "Harrer",
    "Gastecker",
    "Garrer",
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
    "X",
    "X",
    "X",
    "X",
    "X",
    "X",
    "X",
    "Reddit",
    "Medium",
    "Facebook",
    "Instagram",
    "YouTube",
    "TikTok",
    "Telegram",
    "Threads",
]

positive_sentences = [
    "This coin is like a rocket ship to the moon, you better hop on! 🚀🌕",
    "The community is so strong, they could crowdfund a moon colony. 🌍🚀",
    "It’s like digital gold, only shinier and smarter. 💰✨",
    "There’s no stopping this one, it's like the internet in the '90s. 🌐📈",
    "It’s so decentralized, even your grandma could run a node. 👵🔗",
    "The roadmap is longer than a Tolstoy novel, and twice as impressive. 📜🚀",
    "You're not just buying a coin, you're buying into the future. 🌟💼",
    "It's the kind of innovation that makes Satoshi smile from wherever he is. 😄💡",
    "If this isn’t the next big thing, nothing is. 🌍💥",
    "You know it’s good when even your taxi driver is asking how to buy some. 🚖💸",
    "It’s so green, it makes Tesla look like a coal mine. 🌱🔋",
    "With this coin, even your pet could retire early. 🐕💰",
    "Hold this one, and you’ll need a second wallet for all the gains. 👜💵",
    "This coin makes DeFi look like child's play. 🧸🔗",
    "It's the most secure chain out there – hackers get scared just looking at it. 🔒🕵️",
    "Even whales are jealous of how big this coin is getting. 🐋📈",
    "It’s like a faucet of passive income – but with jets. 💸💨",
    "If it were any more decentralized, it would dissolve into thin air. 🌀🔗",
    "This is the Lamborghini of cryptocurrencies. 🏎️💸",
    "You can feel the gains just by holding it in your wallet. 💼💹",
    "This is the one coin that makes hodling feel like winning the lottery! 🎉💎",
    "Get ready to see gains that make Wall Street jealous. 💹💸",
    "It’s like owning a slice of the future – and it’s only going up from here. 🚀📈",
    "The team behind this coin could probably launch a spaceship if they wanted to. 🚀🛠️",
    "It’s more stable than your 9-to-5, with triple the upside. 💼📈",
    "This project is making more partnerships than a Fortune 500 company! 🤝💼",
    "It’s as if someone took the best ideas in crypto and wrapped them into one coin. 💡🔗",
    "You can almost feel the innovation pulsing through the blockchain. 🔋💥",
    "Even skeptics are starting to say, ‘Maybe I should get in on this.’ 💭📈",
    "This is the kind of project that could make early adopters legends. 🏆👑",
    "When in doubt, zoom out – the trajectory of this coin is sky-high. 📈🌌",
    "Forget holding cash – this coin is the real safe haven. 🏦🔒",
    "The dev team is so transparent, it feels like watching magic happen in real-time. 🔍✨",
    "It’s the kind of investment that’ll have your grandkids saying, ‘You were there when?’ 👶💰",
    "Buy a little now, thank yourself a lot later. 🛒💰",
    "It’s the gold rush all over again, but this time, it's digital. 💰⛏️",
    "New era, new rules – this coin plays by its own. 🎮🚀",
    "The roadmap looks like a blueprint for disrupting the world economy. 📜💡",
    "If you’ve got diamond hands, this coin is a gift. 💎🎁",
    "The community spirit here could power a whole city. 🌆💡",
    "It’s the kind of breakthrough that’ll make traditional finance look ancient. 🏦📈",
    "The only thing outpacing the tech is the community support. 🫂📈",
    "You can practically feel the gains compounding as you watch. 📈💰"
]

negative_sentences = [
    "This coin’s future is as bright as a broken lightbulb. 💡💔",
    "It’s about as decentralized as a pizza shop. 🍕🔒",
    "It’s tanking faster than a lead balloon. 🎈📉",
    "If you’re looking for a quick way to lose money, this is it. 💸⏳",
    "The roadmap looks like a treasure map… but without the treasure. 🗺️❌",
    "This is the kind of coin that even rug pulls don't bother with. 🧹🚫",
    "It’s so volatile, it gives rollercoasters a run for their money. 🎢💹",
    "Investing in this coin is like throwing money into a black hole. 🕳️💸",
    "It’s a ghost town – even the developers have ghosted. 👻🔗",
    "The only thing this coin is mining is disappointment. ⛏️💔",
    "It's dropping faster than a stone tied to an anchor. ⚓📉",
    "You'd be better off investing in rocks – at least they don't move. 🪨💸",
    "It’s been outperformed by actual memes. 😂📉",
    "Even your toaster has more utility than this coin. 🍞🔌",
    "It’s not just going to zero, it’s digging a hole past it. 🕳️0️⃣",
    "This coin's security is so bad, it’s practically an invitation to hackers. 🔓🕵️",
    "Buying this coin is like trying to catch a falling knife. 🔪📉",
    "The only roadmap here is the one to bankruptcy. 📜💸",
    "If vaporware had a mascot, it’d be this coin. 🌀🤖",
    "It's a hype train with no brakes and no destination. 🚂💔",
    "Trying to find value in this coin is like looking for water in the desert. 🏜️💸",
    "The only thing this project is mining is people’s hopes. ⛏️💤",
    "Investing in this feels like watching sand slip through your fingers. 🏖️💸",
    "This coin is about as transparent as a brick wall. 🧱💡",
    "The only thing it’s disrupting is your wallet. 💸💀",
    "Promises are big, but delivery is about as reliable as a paper umbrella. ☔❌",
    "Even the support team seems to be ghosting at this point. 👻🤦",
    "You'd be better off stashing your cash under your mattress. 🛏️💵",
    "The only thing high about this coin is the risk. ⚠️📉",
    "It’s the Titanic of crypto, and it's already hit the iceberg. 🚢🧊",
    "If you like playing with fire, this coin’s perfect. 🔥💸",
    "Its whitepaper has more fantasy than a sci-fi novel. 📜👽",
    "Trusting this coin is like trusting a fox to guard the henhouse. 🦊🐔",
    "It’s so illiquid, you’d have a better chance cashing out with seashells. 🐚💸",
    "If it were any shakier, it’d need a warning label. ⚠️📉",
    "Trying to ‘hodl’ this feels like sitting on a ticking time bomb. ⏳💥",
    "The roadmap only leads to one place: disappointment. 🗺️💔",
    "Just remember, even hot air balloons eventually come down. 🎈📉",
    "It’s like burning cash in a furnace – but less warm. 🔥💸",
    "Watching this coin drop is like watching a slow-motion train wreck. 🚂💥",
    "If you've hit rock bottom, this coin is ready to dig deeper. 🪨📉",
    "The security here is about as strong as JavaScript’s type checking – good luck. 🔓💀",
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
    return 1, sentiment, f"{name} posted on {platform}: " + " ".join(sentences)


def get_market_event(eiq):
    if eiq:
        return eiq.pop(0)
    event_generators = [generate_cryptocurrency_opinion]
    return random.choice(event_generators)()


def _interpolate(start, end, t):
    """Linear interpolation between two points."""
    if not -0.0001 <= t <= 1.0001:
        raise ValueError("cannot interpolate with t outside of [0, 1]")
    SIGMOID_SQUISH = 4
    s = 1 / (1 + math.exp(SIGMOID_SQUISH * (1 - 2 * t)))
    return start * (1 - s) + end * s
    # return start + (end - start) * t


class MarketSim:
    def __init__(
        self,
        base_value=0.0,
        bounce_back_value=0.0,
        event_decay_factor=0.5,
        market_boost_increase_factor=1.0,
        stddev=1.0,
        seed=None,
        freqs: Optional[Union[list, int]] = 8,
        extra_freq_weights=None,
        event_impact=1.0,
        event_boost_weight=1,
        event_prob=0.05,
        event_change_trend_min_stds_from_mean=1,
        event_change_trend_weight=1.0,
    ):
        if freqs is None:
            freqs = 8
        if isinstance(freqs, int):
            freqs = [2**i for i in range(freqs)]
        # Initialize market state
        self.market_boost = 1
        self.bounce_back_value = bounce_back_value
        self.market_boost_increase_factor = market_boost_increase_factor
        self.price = 0
        self.event_boost = 0.0
        self.decay_factor = event_decay_factor
        self.event_boost_weight = event_boost_weight
        self.event_change_trend_min_stds_from_mean = (
            event_change_trend_min_stds_from_mean
        )
        self.base_value = base_value
        self.event_change_trend_weight = event_change_trend_weight
        self.stddev = stddev
        self.seed = seed
        self.extra_freq_weights = extra_freq_weights
        self.event_prob = event_prob
        self.event_impact = event_impact
        if seed is not None:
            random.seed(seed)

        # Store noise values for different frequencies
        self.max_freq = max(freqs)
        self.noise_frequencies = [
            1 / i for i in freqs
        ]  # Frequencies: 1, 1/2, 1/4, ..., 1/max_freq
        self.inverse_frequencies = freqs[:]  # Time steps between samples
        self.noise_values = {
            freq: [random.gauss(0, self.stddev), random.gauss(0, self.stddev)]
            for freq in self.noise_frequencies
        }
        self.step_counter = 1

    def _get_fractal_noise(self):
        total_noise = 0.0

        # Sum up interpolated noise values from each frequency, weighted by 1/frequency
        for i, freq, inv_freq in zip(
            range(len(self.noise_frequencies)),
            self.noise_frequencies,
            self.inverse_frequencies,
        ):
            # Update noise for next step if we've hit the next sample point
            if self.step_counter % inv_freq == 0:
                self.noise_values[freq][0] = self.noise_values[freq][1]
                self.noise_values[freq][1] = random.gauss(0, self.stddev)

            alpha = (self.step_counter % inv_freq) / inv_freq

            start_noise = self.noise_values[freq][0]
            end_noise = self.noise_values[freq][1]
            interpolated_noise = _interpolate(start_noise, end_noise, alpha)

            total_noise += (
                interpolated_noise
                * inv_freq
                * (1 if self.extra_freq_weights is None else self.extra_freq_weights[i])
            )

        # Scale noise by market boost
        total_noise *= self.market_boost

        return total_noise

    def step(self, eiq):
        # Continue generating noise until the price is above bounce-back value
        while True:
            noise = self._get_fractal_noise()

            self.price = (
                noise + self.event_boost * self.event_boost_weight + self.base_value
            )

            if self.price > self.bounce_back_value:
                self.event_boost *= self.decay_factor
                self.market_boost *= self.market_boost_increase_factor
                break
            else:
                self._resample_all_noise()

        # Handle events
        if random.random() < self.event_prob:
            mag, sentiment, message = get_market_event(eiq)
            self._handle_event(mag, sentiment)
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
            self.noise_values[freq][1] = (
                abs(random.gauss(0, self.stddev))
                + self.bounce_back_value
                - self.base_value
            ) / sum(a * b for a, b in zip(self.inverse_frequencies, self.extra_freq_weights))
            self.event_boost = 0
            self.market_boost = 1
        self.step_counter = 0

    def _handle_event(self, mag, sentiment):
        # Generate a random boost/damp value for events
        sentiment_sign = 1 if sentiment == "positive" else -1
        event_impact = abs(random.gauss(0, 1))
        self.event_boost += event_impact * self.event_impact * sentiment_sign * mag

        # note that std of event_impact distribution is 1
        if event_impact > self.event_change_trend_min_stds_from_mean:
            for i in range(len(self.noise_frequencies)):
                freq = self.noise_frequencies[i]
                inv_freq = self.inverse_frequencies[i]
                noise_values = self.noise_values[freq]
                noise_values[0] = _interpolate(
                    noise_values[0],
                    noise_values[1],
                    (self.step_counter % inv_freq) * freq,
                )
                noise_values[1] += (
                    event_impact
                    * self.event_change_trend_weight
                    * sentiment_sign
                    * self.stddev
                )
            self.step_counter = 1


class MarketSimMix:
    def __init__(self, sims: tuple["MarketSim", ...], event_inject_queues: tuple[Optional[list], ...]):
        assert len(sims) == len(event_inject_queues)
        self.sims: Tuple["MarketSim", ...] = sims
        self.event_inject_queues = event_inject_queues
        self.price = sum(sim.base_value for sim in sims)

    def step(self):
        net_price = 0
        out_event = None
        for sim, eiq in zip(self.sims, self.event_inject_queues):
            price, event = sim.step(eiq)
            net_price += price
            if event is not None:
                out_event = event  # TODO this is very hacky since it should show the most important event, but should be enough for my specific use of this class
        self.price = net_price
        return net_price, out_event


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    vs = []
    window_size = 1000  # Number of points to display in the window
    sim = MarketSim(
        base_value=30.0,
        bounce_back_value=10.0,
        market_boost_increase_factor=1.0,
        seed=42,
        stddev=0.04,
        freqs=[1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024],
        extra_freq_weights=[1, 1, 1, 1, 1, 1, 1, 1, 1 / 2, 1 / 4, 1 / 8],
        event_decay_factor=0.998,
        event_impact=0.001,
        event_boost_weight=100,
        event_prob=0.01,
        event_change_trend_min_stds_from_mean=1,
        event_change_trend_weight=0.8,
    )

    # Prepare the figure and axis
    fig, ax = plt.subplots()
    (line,) = ax.plot([])  # Empty plot initially
    ax.set_xlim(0, window_size)  # Fixed window size of 1000 points

    def init():
        """Initialize the background of the plot."""
        line.set_data([], [])
        return (line,)

    def update(frame):
        """Update the plot with the next price."""
        # Get the next price from the simulation
        for i in range(10):
            price = sim.step()[0]
            vs.append(price)

            # Show only the last `window_size` points in the plot
            if len(vs) > window_size:
                current_data = vs[-window_size:]
            else:
                current_data = vs

        # Update the x and y data for the plot
        line.set_data(range(len(current_data)), current_data)
        ax.set_ylim(0, max(current_data) + 0.1)  # Adjust y-limits dynamically
        return (line,)

    # Create the animation
    ani = FuncAnimation(
        fig, update, frames=range(1000), init_func=init, blit=False, interval=50
    )

    ani.save("market_simulation.mp4", writer="ffmpeg", fps=30)
    # plt.show()
