import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from black_scholes import black_scholes_price, calculate_greeks
from monte_carlo import monte_carlo_option_price, plot_price_paths
from binomial_tree import binomial_tree_option
from barrier_option import monte_carlo_barrier_option

st.set_page_config(page_title="🧪 Option Pricing Lab", layout="wide")

st.markdown("""
    <style>
    section[data-testid="stSidebar"] > div:first-child {
        padding-top: 0rem !important;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🧪 Option Pricing Lab")
st.caption("Switch between pricing models, simulations, and learning tabs below.")

tab1, tab2, tab3 = st.tabs(["🔢 Price an Option", "📈 Simulate Price Paths", "📘 Learn the Basics"])

# Sidebar Inputs
with st.sidebar:
    st.header("🎛️ Global Settings")
    option_type = st.selectbox("📝 What kind of option?", ["call", "put"])
    S = st.slider("📈 Stock Price Now", 50, 150, 100)
    K = st.slider("🎯 Strike Price", 50, 150, 100)
    T = st.slider("⏳ Time to Expiry (days)", 1, 365, 30) / 365
    r = st.slider("🏦 Risk-Free Rate (%)", 0.0, 10.0, 5.0) / 100
    sigma = st.slider("🌪️ Volatility (%)", 1.0, 100.0, 20.0) / 100

# Tab 1: Pricing Models
with tab1:
    st.subheader("🔢 Option Pricing Models")
    model = st.radio("Choose a pricing model", [
        "Black-Scholes", "Binomial Tree", "Monte Carlo", "Asian Option", "Barrier Option"
    ])

    if model == "Black-Scholes":
        price = black_scholes_price(S, K, T, r, sigma, option_type)
        st.metric("💰 Option Price", f"${price:.2f}")
        greeks = calculate_greeks(S, K, T, r, sigma, option_type)
        with st.expander("📘 Greeks"):
            st.write(greeks)

    elif model == "Binomial Tree":
        steps = st.slider("🔁 Tree Steps", 10, 500, 100)
        price = binomial_tree_option(S, K, T, r, sigma, N=steps, option_type=option_type)
        st.metric("🌲 Tree Price", f"${price:.2f}")

    elif model == "Monte Carlo":
        steps = st.slider("📊 Time Steps", 10, 500, 100)
        simulations = st.slider("🎲 Simulations", 1000, 200000, 100000, step=1000)
        price, ci = monte_carlo_option_price(S, K, T, r, sigma, simulations, steps, option_type, asian=False)
        st.metric("🎲 Monte Carlo Price", f"${price:.2f}")
        st.caption(f"95% CI: [{ci[0]:.2f}, {ci[1]:.2f}]")

        # Strategy Breakdown
        contract_size = 100
        total_cost = round(price * contract_size, 2)
        break_even = round(K + price if option_type == "call" else K - price, 2)
        st.markdown(f'''
#### 💡 Strategy Breakdown
- 💰 **Total Cost**: `${total_cost}`
- 🎯 **Break-Even Point**: `${break_even}`
- 📉 **Max Loss**: `${total_cost}`
- 📈 **Max Profit**: `Unlimited (calls)` or `Up to ${K} (puts)`
''')

    elif model == "Asian Option":
        steps = st.slider("📊 Time Steps", 10, 500, 100)
        simulations = st.slider("🎲 Simulations", 1000, 200000, 100000, step=1000)
        price, ci = monte_carlo_option_price(S, K, T, r, sigma, simulations, steps, option_type, asian=True)
        st.metric("🍱 Asian Option Price", f"${price:.2f}")
        st.caption(f"95% CI: [{ci[0]:.2f}, {ci[1]:.2f}]")

    elif model == "Barrier Option":
        steps = st.slider("📊 Time Steps", 10, 500, 100)
        simulations = st.slider("🎲 Simulations", 1000, 200000, 100000, step=1000)
        barrier = st.slider("🚧 Barrier Level", 50, 150, 110)
        barrier_type = st.selectbox("🪟 Barrier Type", [
            "up-and-out", "up-and-in", "down-and-out", "down-and-in"
        ])
        price = monte_carlo_barrier_option(S, K, T, r, sigma, barrier, barrier_type, option_type, simulations, steps)
        st.metric("🚧 Barrier Option Price", f"${price:.2f}")

# Tab 2: Simulation Visuals
with tab2:
    st.subheader("📈 Price Path Simulator")
    sim_type = st.radio("What type of paths to simulate?", ["Monte Carlo (Vanilla)", "Asian", "Barrier-aware"])
    steps = st.slider("📊 Steps", 10, 500, 100, key="sim_steps")
    if sim_type == "Barrier-aware":
        barrier = st.slider("🚧 Barrier Level", 50, 150, 110, key="barrier_path")
        plot_price_paths(S, T, r, sigma, 10, steps, barrier=barrier, highlight_barrier=True)
    else:
        plot_price_paths(S, T, r, sigma, 10, steps)

# Tab 3: Learning Tab
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
