import numpy as np

def binomial_tree_option(S, K, T, r, sigma, N=100, option_type="call", american=False):
    dt = T / N
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    p = (np.exp(r * dt) - d) / (u - d)
    discount = np.exp(-r * dt)

    ST = np.array([S * u**j * d**(N - j) for j in range(N + 1)])
    option = np.maximum(ST - K, 0) if option_type == "call" else np.maximum(K - ST, 0)

    for i in range(N - 1, -1, -1):
        ST = np.array([S * u**j * d**(i - j) for j in range(i + 1)])
        option = discount * (p * option[1:i + 2] + (1 - p) * option[:i + 1])

        if american:
            intrinsic = ST - K if option_type == "call" else K - ST
            option = np.maximum(option, intrinsic)

    return option[0]
