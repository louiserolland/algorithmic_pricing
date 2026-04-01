"""
D300 Causal Inference and Machine Learning — Simulation Pipeline
================================================================
Replicates Calvano et al. (2020) DGP, calibrated to Assad et al. (2024).

An environment is created where we have different type of firms that compete
between each other on price.
There are two types of firms:
- Q-learning algorthims which represent algorithmic pricing
- Myopic best-response which represent human pricing
(For this type of situation, firms base their price on current conditions.
They are not thinking startegically about the future, hence the myopic term.
It simplifies human decision by stating that they won't anticipate future consequences))


The goal of this simulated data is to generate observational data while knowing the ground truth
so that we can assess how well the causal estimators recover the true treatment efffect
of algorithmic pricing on profits.

Treatment  D=1: both firms use Q-learning (algorithmic pricing)
Treatment  D=0: both firms use myopic best-response (human pricing)
Confounding:   adoption P(D=1) is correlated with market structure (n, mu, delta)
Instrument  Z: HQ adoption rate (fraction of same-chain OTHER markets that adopted)
Outcome     Y: profit gain Delta = (pi_avg - pi_Nash) / (pi_monopoly - pi_Nash)


Treatment:

We treat the treatment as binary. The competitive environment is either fully 
algorithmic or fully human. This comes from Assad et al. (2024) paper where the
3rd category: human - algorithm has messier results. This also comes from Calvano's paper
where he uses symmetric firms to find results.


Cofounding:

In order to introduce cofounding in the data, we tried to replicate real world situation.
It seems that markets don't adapt algorithms randomly, therefore an adoption probability was given.
The function results are the following:

- when there are more firms, firms adopt algorithm more often because the competitive pressure is probably higher.
This is the result of Bertrand competition when there are more symmetric firms.

- when there is more differentiation (higher mu), firms adopt less often because they can maintain higher prices without algorithms.
It means that products are already strongly differentiated therefore, demand becomes less elastic with respect to price differences.
Firms already have pricing power from product differentiation alone.

- when firms are more patient (higher delta), they adopt less often because they can collude without algorithms.
When delta is low, firms are impatient and have short time horizons.
There are no human tacit collusion as future punishment don't impact as much the price they will choose today.
This is the situation where algorithms add the most value.

- chain_shock is an unobserved HQ-level tendency to adopt.

The confounding problems is that these covariates also directly impact the profit.
There are market structures driving algorithmic adoption and we want to see 
if our ML methods will be able to recover those causal structures.

The instrument used in the analysis finds a source of variation in adoption that is unrelated to 
the market's own competitive structure.
The intrument comes from Assad et al (2024)




Output: simulation_data_v4.csv — ready for R analysis

Usage:
    python simulatev4.py            # very long run
    python simulatev4.py --fast     # Quick test 
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
    Q-learning agents following Calvano et al. (2020) exactly.

    Key details matching the paper:
    - m = 15 price grid points (Calvano baseline)
    - beta in [0, 2e-5]; our grid uses beta = 4e-6 (low exploration, more thorough)
    - Q_0 initialised to discounted payoff under opponent uniform randomisation
      (equation 8 in Calvano): Q_i0(s,a) = sum_{a_{-i}} pi_i(a,a_{-i}) / ((1-delta)*|A|^{n-1})
    - Convergence criterion: policy (argmax Q over actions, per state) unchanged
      for 100,000 consecutive periods — not a fixed iteration count.
    - Maximum iterations: n_iter (cap to avoid infinite loops)
    - Delta recorded over last 100,000 periods after convergence or at end.

    Returns: (Delta, avg_price, converged)
        Delta = (avg_pi - pi_N) / (pi_M - pi_N)  in [0,1]
    """
    rng = np.random.default_rng(seed)

    c = np.full(n, c_val)
    a = np.full(n, a_i)

    # Benchmark prices
    p_M = monopoly_price(c_val, a_i, mu, a0, n)
    p_N = nash_price(c_val, a_i, mu, a0, n)

    pi_M = np.mean(compute_profit(np.full(n, p_M), c, a, mu, a0))
    pi_N = np.mean(compute_profit(np.full(n, p_N), c, a, mu, a0))

    if abs(pi_M - pi_N) < 1e-10:
        return 0.0, p_N, False

    # Price grid: m equally spaced points in [p_N - xi*(p_M-p_N), p_M + xi*(p_M-p_N)]
    xi = 0.1
    p_grid = np.linspace(p_N - xi * (p_M - p_N),
                         p_M + xi * (p_M - p_N), m)

    # State space: rivals' last prices, encoded as integer in base m
    n_states = m ** n

    # --- Calvano Q initialisation (equation 8) ---
    # Q_i0(s, a_i) = E[pi_i | a_i, a_{-i} ~ Uniform] / (1 - delta)
    # = sum_{a_{-i}} pi_i(a_i, a_{-i}) / ((1-delta) * m^{n-1})
    Q = []
    for i in range(n):
        Q_i = np.zeros((n_states, m))
        for ai_idx in range(m):
            # Average profit for firm i when it plays ai_idx and rivals randomise
            total_pi = 0.0
            count = 0
            for rival_actions in product(range(m), repeat=n - 1):
                prices = np.array([p_grid[ai_idx]] +
                                  [p_grid[r] for r in rival_actions])
                if i > 0:
                    prices = np.insert(np.delete(prices, 0), i, prices[0])
                pi = compute_profit(prices, c, a, mu, a0)
                total_pi += pi[i]
                count += 1
            avg_pi = total_pi / count / (1.0 - delta)
            Q_i[:, ai_idx] = avg_pi   # same for all states (symmetric init)
        Q.append(Q_i)

    # State encoding: price_indices -> integer in base m
    def encode(indices):
        s = 0
        for idx in indices:
            s = s * m + int(idx)
        return s

    # Initialise with random state
    cur_actions = rng.integers(0, m, size=n)
    cur_state   = encode(cur_actions)

    # --- Convergence tracking (Calvano criterion) ---
    # Check every check_freq steps; require stable_needed consecutive stable checks
    check_freq    = 10_000          # check policy every 10k steps
    stable_target = 10              # 10 × 10k = 100k consecutive stable periods
    stable_count  = 0
    last_policy   = [np.argmax(Q[i], axis=1).copy() for i in range(n)]

    # Rolling profit window for Delta (last 100k periods)
    window_size   = 100_000
    profits_ring  = np.zeros(window_size)
    ring_ptr      = 0
    ring_filled   = False

    converged     = False
    prices_final  = np.zeros(n)

    for t in range(n_iter):
        eps = np.exp(-beta_exp * t)

        # Choose actions (epsilon-greedy; tie-break by lowest price index)
        new_actions = np.array([
            rng.integers(0, m) if rng.random() < eps
            else np.argmax(Q[i][cur_state])
            for i in range(n)
        ])

        prices  = p_grid[new_actions]
        pi      = compute_profit(prices, c, a, mu, a0)
        new_state = encode(new_actions)

        for i in range(n):
            best_next = np.max(Q[i][new_state])
            Q[i][cur_state, new_actions[i]] += alpha_lr * (
                pi[i] + delta * best_next - Q[i][cur_state, new_actions[i]]
            )

        # Rolling profit window
        profits_ring[ring_ptr % window_size] = np.mean(pi)
        ring_ptr += 1
        if ring_ptr >= window_size:
            ring_filled = True

        cur_state   = new_state
        cur_actions = new_actions

        # Convergence check every check_freq iterations
        if (t + 1) % check_freq == 0:
            new_policy = [np.argmax(Q[i], axis=1) for i in range(n)]
            policy_stable = all(
                np.array_equal(new_policy[i], last_policy[i]) for i in range(n)
            )
            if policy_stable:
                stable_count += 1
                if stable_count >= stable_target:
                    converged = True
                    prices_final = prices
                    break
            else:
                stable_count = 0
                last_policy = [p.copy() for p in new_policy]

    # Compute Delta from rolling window
    if ring_filled:
        avg_pi = float(np.mean(profits_ring))
    elif ring_ptr > 0:
        avg_pi = float(np.mean(profits_ring[:ring_ptr]))
    else:
        avg_pi = pi_N

    Delta = float(np.clip((avg_pi - pi_N) / (pi_M - pi_N), 0.0, 1.0))

    if not converged:
        prices_final = prices

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

def adoption_prob(n, mu, delta, gamma, chain_shock=0.0):
    """
    P(D=1 | n, mu, delta, chain_shock) — logistic adoption model.

    Economic motivation (calibrated to Assad et al. 2024):
      - More firms (n) → higher adoption (more competitive pressure)
      - Higher mu (more differentiation) → lower adoption
      - Higher delta (more patient) → lower adoption
        (patient firms can collude without algorithms)
      - chain_shock: unobserved HQ-level tendency to adopt (~ N(0, sigma_chain))
        This creates within-chain correlation in D beyond what local X explains,
        giving Z_hq (leave-one-out chain adoption rate) genuine first-stage power.

    gamma = (intercept, gamma_n, gamma_mu, gamma_delta)
    Calibrated so duopoly adopts ~35%, triopoly ~60%.
    """
    g0, g_n, g_mu, g_delta = gamma
    log_odds = (g0 + g_n * (n - 2) + g_mu * (mu - 0.25)
                - g_delta * (delta - 0.95) + chain_shock)
    return float(1.0 / (1.0 + np.exp(-log_odds)))


# ============================================================
# SECTION 5: SINGLE SESSION RUNNER  (multiprocessing target)
# ============================================================

def run_session(args):
    """
    Run one simulation session. Returns a results dict.
    Designed to be called via multiprocessing.Pool.map.

    delta_obs / mu_obs are the jittered values used for the actual simulation
    and stored as covariates in the output CSV.  delta_grid / mu_grid are the
    original grid values, kept only to identify the chain (for Z_hq) and are
    NOT stored — the R analysis never sees them.
    """
    (session_id, n,
     delta_obs, mu_obs,
     alpha_lr, beta_exp,
     c_val, a_i, a0, m_prices,
     n_iter_ql, n_iter_br,
     D, chain_id, seed) = args

    if D == 1:
        Delta, avg_price, converged = simulate_qlearning(
            n=n, delta=delta_obs, mu=mu_obs,
            alpha_lr=alpha_lr, beta_exp=beta_exp,
            c_val=c_val, a_i=a_i, a0=a0,
            m=m_prices, n_iter=n_iter_ql, seed=seed
        )
    else:
        Delta, avg_price = simulate_myopic(
            n=n, mu=mu_obs, c_val=c_val, a_i=a_i,
            a0=a0, n_iter=n_iter_br, seed=seed
        )
        converged = True  # myopic always converges

    return {
        'session_id': session_id,
        'chain_id':   chain_id,
        'n':          n,
        'delta':      delta_obs,
        'mu':         mu_obs,
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
    sigma_chain = config.get('chain_shock_sd', 0.8)

    # Jitter configuration — adds continuous variation so GRF can find
    # meaningful splits rather than acting on a pure step function.
    # Grid values are kept for chain_id assignment (instrument validity);
    # jittered values are what the DGP actually runs on and what R sees.
    jitter = config.get('jitter_sd', {'delta': 0.0, 'mu': 0.0, 'alpha': 0.0})

    # Random chain assignment: each covariate cell maps to one of n_chains chains.
    # Chains cross the covariate space, so within any forest leaf observations
    # come from multiple chains — giving Z_hq genuine residual variation after
    # conditioning on X. This is the fix for the original within-leaf weak-IV problem.
    n_chains = config.get('n_chains', 10)
    all_cells = list(product(n_vals, delta_vals, mu_vals, alpha_vals))
    rng_assign = np.random.default_rng(config['master_seed'] + 99)
    chain_id_map = {cell: int(rng_assign.integers(0, n_chains))
                    for cell in all_cells}

    # Continuous HQ-level shock per chain ~ N(0, sigma_chain).
    # Economically: HQ-level adoption culture / procurement tendency varies
    # continuously across brands — mirrors Assad et al.'s network diffusion story.
    # The shock shifts adoption log-odds but does NOT enter profit equations
    # -> exclusion restriction holds.
    chain_shocks = {c: float(rng.normal(0, sigma_chain))
                    for c in range(n_chains)}

    sessions = []
    session_id = 0

    for s in range(S):
        for n, delta, mu, alpha_lr in product(n_vals, delta_vals, mu_vals, alpha_vals):

            delta_obs = float(np.clip(
                delta + rng.normal(0, jitter.get('delta', 0.0)),
                0.50, 0.99
            ))
            mu_obs = float(np.clip(
                mu + rng.normal(0, jitter.get('mu', 0.0)),
                0.05, 1.0
            ))
            alpha_obs = float(np.clip(
                alpha_lr + rng.normal(0, jitter.get('alpha', 0.0)),
                0.01, 0.50
            ))

            cid = chain_id_map[(n, delta, mu, alpha_lr)]
            p = adoption_prob(n, mu_obs, delta_obs, gamma,
                              chain_shock=chain_shocks[cid])
            D = int(rng.random() < p)

            seed = int(rng.integers(0, 2**31))

            sessions.append((
                session_id, n,
                delta_obs, mu_obs,
                alpha_obs, config['beta_exp'],
                config['c_val'], config['a_i'], config['a0'],
                config['m_prices'],
                config['n_iter_ql'], config['n_iter_br'],
                D, cid, seed
            ))
            session_id += 1

    return sessions


def add_instrument(df):
    """
    Construct leave-one-out HQ adoption instrument (Z_hq).

    For each observation i in chain c:
        Z_i = mean(D_j) for j in same chain, j != i

    Validity: Z is correlated with D via the chain-level adoption shock
    (captured in chain_shocks), but excluded from the profit equation
    conditional on local X — exclusion restriction holds by construction.

    Chains are randomly assigned across covariate cells (not defined by X),
    so Z_hq retains genuine variation after conditioning on X in any forest leaf.
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
    print("\n  Instrument validity — corr(Z_hq, D):")
    print(f"  {df['Z_hq'].corr(df['D']):.3f}  (should be > 0.15 with random chains)")

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
            beta_exp   = 4e-6,            # within Calvano range [0, 2e-5]
            n_sessions = 3,

            # Calvano baseline parameters (exact match)
            c_val    = 1.0,
            a_i      = 2.0,               # a_i - c_i = 1, so a_i = 2.0
            a0       = 0.0,
            m_prices = 15,                # Calvano's exact value (was 11)

            # Max iterations per session — convergence criterion stops early.
            # Calvano reports ~400k-2M periods depending on beta.
            # With beta=4e-6 (low exploration) expect ~600k-1.2M.
            n_iter_ql = 2_000_000,
            n_iter_br = 200,

            gamma          = (-0.6, 1.2, -2.0, 1.5),
            n_chains       = 10,
            chain_shock_sd = 1.5,
            jitter_sd      = {'delta': 0.03, 'mu': 0.02, 'alpha': 0.02},

            n_workers   = max(1, cpu_count() - 1),
            master_seed = 42,
        )
        outfile = 'simulation_data_fast.csv'

    else:
        # Full overnight run — Calvano-calibrated parameters
        # Expected runtime: 6-12 hours on a modern laptop (multiprocessing)
        # ~1080 sessions; ~60% Q-learning; convergence typically at 600k-1.5M iters.
        config = dict(
            # Parameter grid
            n_vals     = [2, 3],
            delta_vals = [0.7, 0.85, 0.95],
            mu_vals    = [0.1, 0.25, 0.5],
            alpha_vals = [0.05, 0.15, 0.25],
            # beta=4e-6 is in Calvano's range [0, 2e-5].
            # Lower beta = more exploration = slower convergence = more thorough.
            # At T=1M: epsilon = exp(-4) ~ 0.018 (nearly converged).
            # At T=2M: epsilon = exp(-8) ~ 0.0003 (fully converged).
            beta_exp   = 4e-6,
            n_sessions = 20,          # sessions per grid cell (1080 total)

            # Calvano baseline parameters — exact match to the paper
            c_val    = 1.0,
            a_i      = 2.0,           # a_i - c_i = 1
            a0       = 0.0,
            m_prices = 15,            # Calvano's exact value (was 11)

            # Max iterations — convergence criterion stops early when policy
            # stabilises for 100,000 consecutive periods (Calvano Section III.B).
            # Cap at 2M to bound runtime; Calvano reports <2M in almost all cases.
            n_iter_ql = 5_000_000,
            n_iter_br = 500,

            gamma          = (-0.6, 1.2, -2.0, 1.5),
            n_chains       = 10,
            chain_shock_sd = 1.5,
            jitter_sd      = {'delta': 0.04, 'mu': 0.03, 'alpha': 0.03},

            n_workers   = max(1, cpu_count() - 1),
            master_seed = 42,
        )
        outfile = 'simulation_data_v4.csv'

    # Validate baseline: check Delta ~ 0.85 for n=2 at Calvano defaults
    print("\nRunning quick baseline sanity check (n=2, delta=0.95, mu=0.25)...")
    print("  (Uses full convergence criterion — may take 1-3 min)")
    Delta_test, _, conv_test = simulate_qlearning(
        n=2, delta=0.95, mu=0.25,
        alpha_lr=0.15, beta_exp=4e-6,
        c_val=1.0, a_i=2.0, a0=0.0,
        m=15, n_iter=2_000_000, seed=0
    )
    print(f"  Baseline Delta = {Delta_test:.3f}  (Calvano target: ~0.85 at baseline)")
    print(f"  Converged: {conv_test}")

    # Run full pipeline
    df = run_pipeline(config)

    df.to_csv(outfile, index=False)
    print(f"\n  Dataset saved to: {outfile}")
    print("  Columns:", list(df.columns))
    print("\nDone. Open analysis.R to run the causal inference analysis.")
