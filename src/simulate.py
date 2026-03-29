"""
D300 Causal Inference and Machine Learning — Simulation Pipeline
================================================================
Replicates Calvano et al. (2020) DGP, calibrated to Assad et al. (2024).

Treatment  D=1: both firms use Q-learning (algorithmic pricing)
Treatment  D=0: both firms use myopic best-response (human pricing)
Confounding:   adoption P(D=1) is correlated with market structure (n, mu, delta)
Instrument  Z: HQ adoption rate (fraction of same-chain OTHER markets that adopted)
Outcome     Y: profit gain Delta = (pi_avg - pi_Nash) / (pi_monopoly - pi_Nash)

Output: simulation_data.csv — ready for R analysis

Usage:
    python simulate.py            # Full overnight run
    python simulate.py --fast     # Quick test (~5 min, small grid)
"""

import sys
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar, brentq
from multiprocessing import Pool, cpu_count
from itertools import product
import time
import warnings
warnings.filterwarnings('ignore')

FAST_MODE = '--fast' in sys.argv

# ============================================================
# SECTION 1: DEMAND AND BENCHMARK PRICES
# ============================================================

def logit_demand(prices, a, mu, a0):
    """
    Logit demand for each firm.
    prices : (n,) array of prices
    a      : (n,) array of quality parameters (a_i)
    mu     : substitutability parameter (>0, lower = less differentiated)
    a0     : outside good utility
    Returns: (n,) array of market shares / quantities
    """
    exp_terms = np.exp((a - prices) / mu)
    denom = np.sum(exp_terms) + np.exp(a0 / mu)
    return exp_terms / denom


def compute_profit(prices, c, a, mu, a0):
    """Per-period profits for all firms."""
    q = logit_demand(prices, a, mu, a0)
    return (prices - c) * q


def monopoly_price(c_val, a_i, mu, a0, n):
    """
    Joint monopoly price (symmetric, all firms charge same p).
    Maximises total industry profit.
    """
    c  = np.full(n, c_val)
    a  = np.full(n, a_i)
    def neg_profit(p):
        return -np.sum(compute_profit(np.full(n, p), c, a, mu, a0))
    res = minimize_scalar(neg_profit, bounds=(c_val + 1e-4, c_val + 5.0), method='bounded')
    return res.x


def nash_price(c_val, a_i, mu, a0, n):
    """
    Bertrand-Nash equilibrium price (symmetric firms).
    FOC for firm i: q_i + (p_i - c_i) * dq_i/dp_i = 0
    For logit: dq_i/dp_i = -q_i*(1 - q_i)/mu
    => p_i = c_i + mu / (1 - q_i)   [implicit, solved numerically]
    """
    c = np.full(n, c_val)
    a = np.full(n, a_i)

    def foc(p_scalar):
        prices = np.full(n, p_scalar)
        q = logit_demand(prices, a, mu, a0)
        dq_dp = -q[0] * (1.0 - q[0]) / mu
        return q[0] + (p_scalar - c_val) * dq_dp

    # Nash price is bracketed between c and monopoly price
    p_M = monopoly_price(c_val, a_i, mu, a0, n)
    try:
        p_N = brentq(foc, c_val + 1e-4, p_M - 1e-4)
    except ValueError:
        p_N = c_val + mu  # fallback
    return p_N


# ============================================================
# SECTION 2: Q-LEARNING SIMULATION  (D = 1)
# ============================================================

def simulate_qlearning(n, delta, mu, alpha_lr, beta_exp,
                        c_val, a_i, a0, m,
                        n_iter, seed):
    """
    Q-learning agents following Calvano et al. (2020).

    State:   each firm observes rivals' last prices (k=1 memory)
             State space size = m^n  (m price grid points, n firms)
    Action:  choose price index from {0, ..., m-1}
    Update:  Q_{t+1}(s,a) = (1-alpha)*Q_t(s,a) + alpha*[pi_t + delta*max_a' Q_t(s',a')]
    Explore: epsilon-greedy, epsilon_t = exp(-beta * t)

    Returns: (Delta, avg_price, converged)
        Delta = (avg_pi - pi_N) / (pi_M - pi_N)  in [0,1]
    """
    rng = np.random.default_rng(seed)

    c = np.full(n, c_val)
    a = np.full(n, a_i)

    # Benchmark prices
    p_M = monopoly_price(c_val, a_i, mu, a0, n)
    p_N = nash_price(c_val, a_i, mu, a0, n)

    # Benchmark profits (symmetric)
    pi_M = np.mean(compute_profit(np.full(n, p_M), c, a, mu, a0))
    pi_N = np.mean(compute_profit(np.full(n, p_N), c, a, mu, a0))

    if abs(pi_M - pi_N) < 1e-10:
        return 0.0, p_N, False

    # Price grid: m equally spaced points in [p_N - xi, p_M + xi]
    xi = 0.1
    p_grid = np.linspace(p_N - xi, p_M + xi, m)

    # Q-tables: shape (n_states, m) per firm
    n_states = m ** n
    Q = [np.zeros((n_states, m)) for _ in range(n)]

    # State encoding: price_indices -> integer in base m
    def encode(indices):
        s = 0
        for i, idx in enumerate(indices):
            s += int(idx) * (m ** i)
        return s

    # Initialise
    cur_actions = rng.integers(0, m, size=n)
    cur_state   = encode(cur_actions)

    # Track profits in convergence window
    window = max(5000, n_iter // 20)
    profits_window = np.zeros((window, n))
    write_ptr = 0

    prices_final = np.zeros(n)

    for t in range(n_iter):
        eps = np.exp(-beta_exp * t)

        # Choose actions (epsilon-greedy)
        new_actions = np.array([
            rng.integers(0, m) if rng.random() < eps
            else np.argmax(Q[i][cur_state])
            for i in range(n)
        ])

        # Realise prices and profits
        prices = p_grid[new_actions]
        pi     = compute_profit(prices, c, a, mu, a0)

        # Transition to new state
        new_state = encode(new_actions)

        # Q-update for each firm
        for i in range(n):
            best_next = np.max(Q[i][new_state])
            Q[i][cur_state, new_actions[i]] += alpha_lr * (
                pi[i] + delta * best_next - Q[i][cur_state, new_actions[i]]
            )

        cur_state   = new_state
        cur_actions = new_actions

        # Record in rolling window
        if t >= n_iter - window:
            profits_window[write_ptr % window] = pi
            write_ptr += 1
            if t == n_iter - 1:
                prices_final = prices

    avg_pi = np.mean(profits_window)
    Delta  = (avg_pi - pi_N) / (pi_M - pi_N)
    Delta  = float(np.clip(Delta, 0.0, 1.0))

    # Simple convergence check: variance in window
    converged = float(np.std(profits_window[:, 0])) < 0.05

    return Delta, float(np.mean(prices_final)), converged


# ============================================================
# SECTION 3: MYOPIC BEST-RESPONSE SIMULATION  (D = 0)
# ============================================================

def simulate_myopic(n, mu, c_val, a_i, a0, n_iter, seed):
    """
    Myopic best-response: each firm maximises static profit given rivals' last price.
    Converges to Bertrand-Nash => Delta ~ 0.

    Returns: (Delta, avg_price)
    """
    rng = np.random.default_rng(seed)

    c = np.full(n, c_val)
    a = np.full(n, a_i)

    p_M = monopoly_price(c_val, a_i, mu, a0, n)
    p_N = nash_price(c_val, a_i, mu, a0, n)

    pi_M = np.mean(compute_profit(np.full(n, p_M), c, a, mu, a0))
    pi_N = np.mean(compute_profit(np.full(n, p_N), c, a, mu, a0))

    # Initialise with random prices
    prices = rng.uniform(p_N * 0.9, p_M * 1.1, size=n)
    prices = np.clip(prices, c_val + 1e-3, None)

    profits_history = []

    for t in range(n_iter):
        new_prices = prices.copy()
        for i in range(n):
            rivals = np.delete(prices, i)

            def neg_profit_i(p_i):
                all_p = np.insert(rivals, i, p_i)
                q = logit_demand(all_p, a, mu, a0)
                return -(p_i - c_val) * q[i]

            res = minimize_scalar(
                neg_profit_i,
                bounds=(c_val + 1e-3, p_M + 1.0),
                method='bounded'
            )
            new_prices[i] = res.x

        prices = new_prices

        if t >= n_iter - 100:
            pi = compute_profit(prices, c, a, mu, a0)
            profits_history.append(np.mean(pi))

    avg_pi = np.mean(profits_history) if profits_history else pi_N

    if abs(pi_M - pi_N) < 1e-10:
        Delta = 0.0
    else:
        Delta = float(np.clip((avg_pi - pi_N) / (pi_M - pi_N), 0.0, 1.0))

    return Delta, float(np.mean(prices))


# ============================================================
# SECTION 4: CONFOUNDING MECHANISM
# ============================================================

def adoption_prob(n, mu, delta, gamma):
    """
    P(D=1 | n, mu, delta) — logistic adoption model.

    Economic motivation (calibrated to Assad et al. 2024):
      - More firms (n) → higher adoption (more competitive pressure)
      - Higher mu (more differentiation) → lower adoption
      - Higher delta (more patient) → lower adoption
        (patient firms can collude without algorithms)

    gamma = (intercept, gamma_n, gamma_mu, gamma_delta)
    Calibrated so duopoly adopts ~35%, triopoly ~60%.
    """
    g0, g_n, g_mu, g_delta = gamma
    log_odds = g0 + g_n * (n - 2) + g_mu * (mu - 0.25) - g_delta * (delta - 0.95)
    return float(1.0 / (1.0 + np.exp(-log_odds)))


# ============================================================
# SECTION 5: SINGLE SESSION RUNNER  (multiprocessing target)
# ============================================================

def run_session(args):
    """
    Run one simulation session. Returns a results dict.
    Designed to be called via multiprocessing.Pool.map.
    """
    (session_id, n, delta, mu, alpha_lr, beta_exp,
     c_val, a_i, a0, m_prices,
     n_iter_ql, n_iter_br,
     D, chain_id, seed) = args

    if D == 1:
        Delta, avg_price, converged = simulate_qlearning(
            n=n, delta=delta, mu=mu,
            alpha_lr=alpha_lr, beta_exp=beta_exp,
            c_val=c_val, a_i=a_i, a0=a0,
            m=m_prices, n_iter=n_iter_ql, seed=seed
        )
    else:
        Delta, avg_price = simulate_myopic(
            n=n, mu=mu, c_val=c_val, a_i=a_i,
            a0=a0, n_iter=n_iter_br, seed=seed
        )
        converged = True  # myopic always converges

    return {
        'session_id': session_id,
        'chain_id':   chain_id,
        'n':          n,
        'delta':      delta,
        'mu':         mu,
        'alpha_lr':   alpha_lr,
        'beta_exp':   beta_exp,
        'D':          D,
        'Delta':      Delta,
        'avg_price':  avg_price,
        'converged':  int(converged),
    }


# ============================================================
# SECTION 6: FULL PIPELINE
# ============================================================

def build_sessions(config):
    """Build the full list of session argument tuples."""
    rng = np.random.default_rng(config['master_seed'])

    n_vals     = config['n_vals']
    delta_vals = config['delta_vals']
    mu_vals    = config['mu_vals']
    alpha_vals = config['alpha_vals']
    gamma      = config['gamma']
    S          = config['n_sessions']

    # Assign chain_ids: each (delta, mu, alpha) triple is a "chain"
    # Markets within a chain share a headquarters -> basis for the IV
    chain_keys = list(product(delta_vals, mu_vals, alpha_vals))
    chain_id_map = {k: i for i, k in enumerate(chain_keys)}

    sessions = []
    session_id = 0

    for s in range(S):
        for n, delta, mu, alpha_lr in product(n_vals, delta_vals, mu_vals, alpha_vals):

            # --- Confounded treatment assignment ---
            p = adoption_prob(n, mu, delta, gamma)
            D = int(rng.random() < p)

            chain_id = chain_id_map[(delta, mu, alpha_lr)]
            seed = int(rng.integers(0, 2**31))

            sessions.append((
                session_id, n, delta, mu, alpha_lr, config['beta_exp'],
                config['c_val'], config['a_i'], config['a0'],
                config['m_prices'],
                config['n_iter_ql'], config['n_iter_br'],
                D, chain_id, seed
            ))
            session_id += 1

    return sessions


def add_instrument(df):
    """
    Construct HQ adoption instrument (Z).

    For each observation i in chain c:
        Z_i = mean(D_j) for j in same chain, j != i
    This is the leave-one-out adoption rate within the chain.

    Validity: Z is correlated with D (chain-level adoption shocks)
    but excluded from the profit equation conditional on local X.
    """
    Z = np.zeros(len(df))
    for idx, row in df.iterrows():
        same_chain = df[(df['chain_id'] == row['chain_id']) & (df.index != idx)]
        if len(same_chain) > 0:
            Z[idx] = same_chain['D'].mean()
        else:
            Z[idx] = row['D']
    df['Z_hq'] = Z
    return df


def run_pipeline(config):
    print(f"\n{'='*60}")
    print("  Calvano DGP Simulation — D300 Project")
    print(f"{'='*60}")

    sessions = build_sessions(config)
    total    = len(sessions)
    print(f"\n  Total sessions : {total}")
    print(f"  CPU workers    : {config['n_workers']}")
    print(f"  Q-learning iter: {config['n_iter_ql']:,}")
    print(f"  Mode           : {'FAST' if FAST_MODE else 'FULL'}")
    print(f"\n  Starting simulation...")
    t0 = time.time()

    with Pool(config['n_workers']) as pool:
        results = pool.map(run_session, sessions)

    elapsed = time.time() - t0
    print(f"\n  Simulation complete in {elapsed/60:.1f} minutes")

    df = pd.DataFrame(results)

    # Add HQ instrument
    print("  Computing HQ adoption instrument...")
    df = add_instrument(df)

    # Summary
    print(f"\n  Dataset shape   : {df.shape}")
    print(f"  Convergence rate: {df['converged'].mean():.1%}")
    print("\n  Mean Delta by (n, D):")
    print(df.groupby(['n', 'D'])['Delta'].mean().round(3).to_string())
    print("\n  Adoption rates by n:")
    print(df.groupby('n')['D'].mean().round(3).to_string())
    print("\n  Instrument validity — corr(Z, D):")
    print(f"  {df['Z_hq'].corr(df['D']):.3f}  (should be > 0.3)")

    return df


# ============================================================
# CONFIGURATION AND ENTRY POINT
# ============================================================

if __name__ == '__main__':

    if FAST_MODE:
        # ~60-120 sessions, runs in ~5-15 min on a laptop
        # Use to verify the pipeline works before overnight run
        config = dict(
            # Parameter grid
            n_vals     = [2, 3],
            delta_vals = [0.7, 0.95],
            mu_vals    = [0.1, 0.25, 0.5],
            alpha_vals = [0.15],
            beta_exp   = 4e-6,
            n_sessions = 3,           # sessions per grid cell

            # Calvano baseline parameters
            c_val    = 1.0,           # marginal cost
            a_i      = 2.0,           # a_i = c_i + 1 = 2.0  (a_i - c_i = 1)
            a0       = 0.0,           # outside good
            m_prices = 11,            # price grid points (Calvano uses 15)

            # Iteration counts
            n_iter_ql = 50_000,       # Q-learning iterations (reduced for speed)
            n_iter_br = 200,          # myopic BR iterations (converges fast)

            # Confounding parameters (logistic)
            # gamma = (intercept, n_coeff, mu_coeff, delta_coeff)
            # Calibrated: duopoly ~35% adoption, triopoly ~60%
            gamma = (-0.6, 1.2, -2.0, 1.5),

            # Compute
            n_workers   = max(1, cpu_count() - 1),
            master_seed = 42,
        )
        outfile = 'simulation_data_fast.csv'

    else:
        # Full overnight run: ~1080 sessions
        # Expected runtime: 4-8 hours on a standard laptop with multiprocessing
        config = dict(
            # Parameter grid
            n_vals     = [2, 3],
            delta_vals = [0.7, 0.85, 0.95],
            mu_vals    = [0.1, 0.25, 0.5],
            alpha_vals = [0.05, 0.15, 0.25],
            beta_exp   = 4e-6,
            n_sessions = 20,          # sessions per grid cell

            # Calvano baseline parameters
            c_val    = 1.0,
            a_i      = 2.0,
            a0       = 0.0,
            m_prices = 11,

            # Iteration counts
            n_iter_ql = 300_000,      # ~Calvano's convergence point for mu=0.25
            n_iter_br = 500,

            # Confounding parameters
            gamma = (-0.6, 1.2, -2.0, 1.5),

            # Compute
            n_workers   = max(1, cpu_count() - 1),
            master_seed = 42,
        )
        outfile = 'simulation_data.csv'

    # Validate baseline: check Delta ~ 0.85 for n=2 at Calvano defaults
    print("\nRunning quick baseline sanity check (n=2, delta=0.95, mu=0.25)...")
    Delta_test, _, _ = simulate_qlearning(
        n=2, delta=0.95, mu=0.25,
        alpha_lr=0.15, beta_exp=4e-6,
        c_val=1.0, a_i=2.0, a0=0.0,
        m=11, n_iter=50_000, seed=0
    )
    print(f"  Baseline Delta = {Delta_test:.3f}  (Calvano target: ~0.85 at full convergence)")
    print("  Note: lower iterations => lower Delta; this is expected in fast mode.\n")

    # Run full pipeline
    df = run_pipeline(config)

    df.to_csv(outfile, index=False)
    print(f"\n  Dataset saved to: {outfile}")
    print("  Columns:", list(df.columns))
    print("\nDone. Open analysis.R to run the causal inference analysis.")
