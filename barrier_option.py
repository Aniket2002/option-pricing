import numpy as np

def monte_carlo_barrier_option(
    S, K, T, r, sigma, barrier, barrier_type="up-and-out",
    option_type="call", simulations=100_000, steps=100
):
    np.random.seed(42)
    dt = T / steps
    Z = np.random.standard_normal((simulations, steps))
    price_paths = np.zeros((simulations, steps + 1))
    price_paths[:, 0] = S

    for t in range(1, steps + 1):
        price_paths[:, t] = price_paths[:, t - 1] * np.exp(
            (r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z[:, t - 1]
        )

    if "up" in barrier_type:
        breached = (price_paths[:, 1:] >= barrier).any(axis=1)
    else:
        breached = (price_paths[:, 1:] <= barrier).any(axis=1)

    active = breached if "in" in barrier_type else ~breached

    ST = price_paths[:, -1]
    payoff = np.where(active, np.maximum(ST - K, 0) if option_type == "call" else np.maximum(K - ST, 0), 0)
    discounted = np.exp(-r * T) * payoff

    return discounted.mean()
