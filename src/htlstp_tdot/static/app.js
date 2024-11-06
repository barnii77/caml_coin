console.log("app.js loaded successfully");

let marketPrices = [];
let candlestickData = [];

let canvas = document.getElementById('cryptoChart');
let ctx = canvas.getContext('2d');
let isCandlestick = false;
let nextMarketStep = -1;
const windowSize = 500;
const barWidthScale = 0.0045;
let candleChartCounter = 5;
const candleChartBatchSize = 5;

// Function to resize canvas dynamically
function resizeCanvas() {
    canvas.width = canvas.offsetWidth;
    canvas.height = canvas.offsetHeight;
}
resizeCanvas();
window.addEventListener('resize', resizeCanvas);

function showCustomAlert(message, type = 'error') {
    const alertBox = document.getElementById("customAlert");
    alertBox.textContent = message;

    // Apply the class for the specific alert type (success, info, or error)
    alertBox.classList.remove('show', 'success', 'info');
    alertBox.classList.add('show', type);

    // Hide the alert after 3 seconds (or however long you want)
    setTimeout(() => {
        alertBox.classList.remove('show');
    }, 10000); // Adjust the timeout for how long the alert should be visible
}

function calculateYScale(data) {
    const minPrice = Math.min(...data.map(d => d.low || d));
    const maxPrice = Math.max(...data.map(d => d.high || d));
    return { minPrice, maxPrice };
}

function drawYAxis(minPrice, maxPrice) {
    const priceStep = (maxPrice - minPrice) / 10; 
    ctx.fillStyle = "#ffffff";
    ctx.font = "12px Arial";
    for (let i = 0; i <= 10; i++) {
        const price = Math.round(minPrice + priceStep * i);
        const y = canvas.height - ((price - minPrice) / (maxPrice - minPrice)) * canvas.height;
        ctx.fillText(price, 10, y);
    }
}

function drawLineChart() {
    resizeCanvas();
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const { minPrice, maxPrice } = calculateYScale(marketPrices);
    drawYAxis(minPrice, maxPrice);

    ctx.beginPath();
    ctx.strokeStyle = "#00ffff";
    ctx.lineWidth = 2;

    marketPrices.forEach((price, index) => {
        let x = (index / (marketPrices.length - 1)) * (canvas.width - 30) + 30;
        let y = canvas.height - ((price - minPrice) / (maxPrice - minPrice)) * canvas.height;

        if (index === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
    });

    ctx.stroke();
}

function drawCandlestickChart() {
    resizeCanvas();
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const { minPrice, maxPrice } = calculateYScale(candlestickData);
    drawYAxis(minPrice, maxPrice);

    let barWidth = (canvas.width - 30) * barWidthScale;
    let spacing = (canvas.width - 30) * (1 - barWidthScale) / candlestickData.length;
    const xOffset = (canvas.width - (spacing * candlestickData.length)) - 5;

    candlestickData.forEach((data, index) => {
        let x = (index * spacing) + xOffset;
        let openY = canvas.height - ((data.open - minPrice) / (maxPrice - minPrice)) * canvas.height;
        let closeY = canvas.height - ((data.close - minPrice) / (maxPrice - minPrice)) * canvas.height;
        let highY = canvas.height - ((data.high - minPrice) / (maxPrice - minPrice)) * canvas.height;
        let lowY = canvas.height - ((data.low - minPrice) / (maxPrice - minPrice)) * canvas.height;

        ctx.beginPath();
        ctx.moveTo(x + barWidth / 2, highY);
        ctx.lineTo(x + barWidth / 2, lowY);
        ctx.strokeStyle = data.close >= data.open ? "#00ff00" : "#ff0000";
        ctx.stroke();

        ctx.fillStyle = data.close >= data.open ? "#00ff00" : "#ff0000";
        ctx.fillRect(x, Math.min(openY, closeY), barWidth, Math.abs(openY - closeY));
    });
}

async function fetchData() {
    if (nextMarketStep === -1) {
        try {
            const response = await fetch("/api/market/next-market-step");
            nextMarketStep = Math.max((await response.json()).next_market_step - windowSize, 0);
        } catch (e) {
            nextMarketStep = -1;
            return;
        }
    }
    try {
        const response = await fetch("/api/market/steps-since/" + nextMarketStep);
        const data = await response.json();
        if (!data.hasOwnProperty('next_market_step')) {
            return;
        }
        nextMarketStep = data.next_market_step;
        if (data.hasOwnProperty("broker_notification")) {
            showCustomAlert(data.broker_notification)
        }
        const ticker = document.getElementById("eventTicker");
        for (const el of data.market_steps) {
            const newPrice = el.exchange_rate;

            if (el.event) {
                ticker.textContent = el.event.message;
                ticker.style.backgroundColor = el.event.sentiment === "positive" ? "#00aa00" : "#aa0000";
            }
            marketPrices.push(newPrice);

            if (marketPrices.length > windowSize) {
                marketPrices.shift();
            }
        }
        candlestickData = [];
        candleChartCounter = (candleChartCounter - data.n_retrieved) % candleChartBatchSize;
        for (let i = candleChartCounter; i < marketPrices.length; i += candleChartBatchSize) {
            const recent = marketPrices.slice(i, i + candleChartBatchSize + 1);
            candlestickData.push({
                open: recent[0],
                high: Math.max(...recent),
                low: Math.min(...recent),
                close: recent[recent.length - 1]
            });
        }
        if (isCandlestick) drawCandlestickChart();
        else drawLineChart();
    } catch (error) {
        console.error("Error fetching exchange rate:", error);
    }
}

// Initial draw
drawLineChart();
setInterval(fetchData, 500); // Fetch new data every 500ms

// Toggle between line and candlestick charts
document.getElementById('lineChartButton').addEventListener('click', () => {
    isCandlestick = false;
    drawLineChart();
});

document.getElementById('candlestickChartButton').addEventListener('click', () => {
    isCandlestick = true;
    drawCandlestickChart();
});

function updateAvailableCoins(coins) {
    document.getElementById('coinsAvailable').textContent = "You have " + coins.toString() + " CC";
}

// Handle Buy and Sell actions
async function handleBuy() {
    const amount = document.getElementById("amount").value;
    try {
        const response = await fetch("/api/buy", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ amount_coins_bought: amount }),
        });
        
        const data = await response.json();
        if (response.ok) {
            showCustomAlert(`Successfully bought ${data.n_coins_bought} CamlCoins!`);
            updateAvailableCoins(data.coins_available);
        } else {
            showCustomAlert(data.error);
        }
    } catch (error) {
        console.error("Error buying coins:", error);
    }
}

async function handleSell() {
    const amount = document.getElementById("amount").value;
    try {
        const response = await fetch("/api/sell", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ amount_coins_sold: amount }),
        });
        
        const data = await response.json();
        if (response.ok) {
            showCustomAlert(`Successfully sold ${data.n_coins_sold} CamlCoins!`);
            updateAvailableCoins(data.coins_available); 
        } else {
            showCustomAlert(data.error);
        }
    } catch (error) {
        console.error("Error selling coins:", error);
    }
}

async function onOpenPosition() {
    const amount = document.getElementById("amount").value;
    try {
        const response = await fetch("/api/broker/open-position", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ leverage: amount }),
        });

        const data = await response.json();
        if (!response.ok) {
            showCustomAlert(data.error);
            return;
        }
    } catch (error) {
        console.error("Error buying coins:", error);
        return;
    }
    document.getElementById("removeOnOpenPosition").style.display = "none";
}

async function onClosePosition() {
    try {
        const response = await fetch("/api/broker/close-position", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({}),
        });

        const data = await response.json();
        if (response.ok) {
            showCustomAlert("You made a total of " + data.net_earnings + " points with this trade!");
        } else {
            showCustomAlert(data.error);
            return;
        }
    } catch (error) {
        console.error("Error buying coins:", error);
        return;
    }
    document.getElementById("removeOnOpenPosition").style.display = "";
}

// Buy and Sell button event listeners
document.getElementById('buyButton').addEventListener('click', handleBuy);
document.getElementById('sellButton').addEventListener('click', handleSell);

document.getElementById('openPosition').addEventListener('click', onOpenPosition);
document.getElementById('closePosition').addEventListener('click', onClosePosition);

// Use stepUp and stepDown for increment/decrement
const amountField = document.getElementById('amount');
document.getElementById('increaseButton').addEventListener('click', () => {
    amountField.stepUp(); // Increments the amount field by 1 step
});

document.getElementById('decreaseButton').addEventListener('click', () => {
    amountField.stepDown(); // Decrements the amount field by 1 step
    if (parseInt(amountField.value) < 1) amountField.value = 1; // Ensures value doesn't go below 1
});

// Ensure the input field only accepts valid numbers and prevents empty values
amountField.addEventListener('input', () => {
    if (isNaN(parseInt(amountField.value)) || amountField.value === '') {
        amountField.value = 1;
    }
});
