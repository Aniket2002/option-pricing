import numpy as np
import matplotlib.pyplot as plt

def monte_carlo_option_price(
    S, K, T, r, sigma, simulations=100_000, steps=1,
    option_type="call", asian=False, confidence_level=0.95
):
    np.random.seed(42)
    dt = T / steps
    Z = np.random.standard_normal((simulations, steps))
    price_paths = np.zeros((simulations, steps + 1))
    price_paths[:, 0] = S

    for t in range(1, steps + 1):
        price_paths[:, t] = price_paths[:, t - 1] * np.exp(
            (r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z[:, t - 1]
        )

    if asian:
        avg_price = price_paths[:, 1:].mean(axis=1)
        payoff = np.maximum(avg_price - K, 0) if option_type == "call" else np.maximum(K - avg_price, 0)
    else:
        ST = price_paths[:, -1]
        payoff = np.maximum(ST - K, 0) if option_type == "call" else np.maximum(K - ST, 0)

    discounted = np.exp(-r * T) * payoff
    mean_price = discounted.mean()
    std_err = discounted.std(ddof=1) / np.sqrt(simulations)

    z_score = 1.96
    ci = (mean_price - z_score * std_err, mean_price + z_score * std_err)

    return mean_price, ci

def plot_price_paths(S, T, r, sigma, simulations=10, steps=50):
    np.random.seed(1)
    dt = T / steps
    Z = np.random.standard_normal((simulations, steps))
    paths = np.zeros((simulations, steps + 1))
    paths[:, 0] = S

    for t in range(1, steps + 1):
        paths[:, t] = paths[:, t - 1] * np.exp(
            (r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z[:, t - 1]
        )

    plt.figure(figsize=(10, 6))
    for i in range(simulations):
        plt.plot(paths[i], lw=1)
    plt.title("Sample Simulated Price Paths (GBM)")
    plt.xlabel("Time Steps")
    plt.ylabel("Stock Price")
    plt.grid(True)
    plt.show()
