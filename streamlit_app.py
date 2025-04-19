import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from black_scholes import black_scholes_price, calculate_greeks
from monte_carlo import monte_carlo_option_price, plot_price_paths
from binomial_tree import binomial_tree_option
from barrier_option import monte_carlo_barrier_option

st.set_page_config(page_title="🧪 Option Pricing Lab", layout="wide")

# Remove top padding in sidebar
st.markdown("""
    <style>
    section[data-testid="stSidebar"] > div:first-child {
        padding-top: 0rem !important;
        margin-top: 0rem !important;
    }
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        margin-top: 0rem !important;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🧪 Option Pricing Lab")
st.caption("Switch between pricing models, simulations, and learning tabs below.")

# Tabs
tab1, tab2, tab3 = st.tabs(["🔢 Price an Option", "📈 Simulate Price Paths", "📘 Learn the Basics"])

# Sidebar inputs
with st.sidebar:
    st.header("🎛️ Global Settings")
    option_type = st.selectbox("📝 What kind of option?", ["call", "put"],
                               help="Call = you win if stock goes up 📈, Put = you win if stock drops 📉")
    S = st.slider("📈 Stock Price Now", 50, 150, 100,
                  help="Current price of the stock in the market.")
    K = st.slider("🎯 Strike Price", 50, 150, 100,
                  help="Price at which you can buy (call) or sell (put) the stock.")
    T = st.slider("⏳ Time to Expiry (days)", 1, 365, 30,
                  help="Days remaining before the option expires.") / 365
    r = st.slider("🏦 Risk-Free Rate (%)", 0.0, 10.0, 5.0,
                  help="Typical return on a safe investment like government bonds.") / 100
    sigma = st.slider("🌪️ Volatility (%)", 1.0, 100.0, 20.0,
                      help="How much the stock price is expected to fluctuate.") / 100

with tab1:
    st.subheader("🔢 Option Pricing Models")

    model = st.radio("Choose a pricing model", [
        "Black-Scholes", "Binomial Tree", "Monte Carlo", "Asian Option", "Barrier Option"
    ])

    if model == "Black-Scholes":
        price = black_scholes_price(S, K, T, r, sigma, option_type)
        greeks = calculate_greeks(S, K, T, r, sigma, option_type)
        st.metric("💰 Option Price", f"${price:.2f}")
        with st.expander("📘 Greeks"):
            st.write(greeks)

    elif model == "Binomial Tree":
        steps = st.slider("🔁 Tree Steps", 10, 500, 100,
                          help="More steps = more accurate tree, but slower to compute.")
        price = binomial_tree_option(S, K, T, r, sigma, N=steps, option_type=option_type)
        st.metric("🌲 Tree Price", f"${price:.2f}")

    elif model == "Monte Carlo":
        steps = st.slider("📊 Time Steps", 10, 500, 100,
                          help="Divides time into steps for each simulated path.")
        simulations = st.slider("🎲 Simulations", 1000, 200000, 100000, step=1000,
                                help="Number of possible future paths to simulate.")
        price, ci = monte_carlo_option_price(S, K, T, r, sigma, simulations, steps, option_type, asian=False)
        st.metric("🎲 Monte Carlo Price", f"${price:.2f}")
        st.caption(f"95% CI: [{ci[0]:.2f}, {ci[1]:.2f}]")

    elif model == "Asian Option":
        steps = st.slider("📊 Time Steps", 10, 500, 100,
                          help="Number of steps used to compute average price.")
        simulations = st.slider("🎲 Simulations", 1000, 200000, 100000, step=1000,
                                help="Higher simulations = smoother results.")
        price, ci = monte_carlo_option_price(S, K, T, r, sigma, simulations, steps, option_type, asian=True)
        st.metric("🍱 Asian Option Price", f"${price:.2f}")
        st.caption(f"95% CI: [{ci[0]:.2f}, {ci[1]:.2f}]")

    elif model == "Barrier Option":
        steps = st.slider("📊 Time Steps", 10, 500, 100,
                          help="Divides the option's life into steps to track barrier crossing.")
        simulations = st.slider("🎲 Simulations", 1000, 200000, 100000, step=1000,
                                help="Number of simulated paths to run.")
        barrier = st.slider("🚧 Barrier Level", 50, 150, 110,
                            help="If stock hits this, option may die or come alive.")
        barrier_type = st.selectbox("🪟 Barrier Type", [
            "up-and-out", "up-and-in", "down-and-out", "down-and-in"
        ], help="Up = barrier above current price, Down = below. Out = dies, In = activates.")

        price = monte_carlo_barrier_option(S, K, T, r, sigma, barrier, barrier_type, option_type, simulations, steps)
        st.metric("🚧 Barrier Option Price", f"${price:.2f}")

with tab2:
    st.subheader("📈 Price Path Simulator")

    sim_type = st.radio("What type of paths to simulate?", ["Monte Carlo (Vanilla)", "Asian", "Barrier-aware"],
                        help="Choose the simulation style for path plotting.")
    steps = st.slider("📊 Steps", 10, 500, 100, key="sim_steps",
                      help="More steps = more realistic path shapes.")
    if sim_type == "Barrier-aware":
        barrier = st.slider("🚧 Barrier Level", 50, 150, 110, key="barrier_path",
                            help="Barrier line will appear on the plot.")
        plot_price_paths(S, T, r, sigma, 10, steps, barrier=barrier, highlight_barrier=True)
    else:
        plot_price_paths(S, T, r, sigma, 10, steps)

with tab3:
    st.subheader("📘 Options 101 – Quick Refresher")
    st.markdown("""
    - **Call Option**: You win if the stock goes 🆙.
    - **Put Option**: You win if the stock goes ⬇️.
    - **Delta**: Change in option value per $1 move in the stock.
    - **Gamma**: How fast Delta changes.
    - **Theta**: Value lost each day (time decay).
    - **Vega**: How much the option reacts to volatility.
    - **Asian Option**: Based on the average price across time.
    - **Barrier Option**: Only activates or dies if price crosses a barrier.
    """)
