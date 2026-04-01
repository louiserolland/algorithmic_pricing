"""
diagnose_simulation.py
======================
Detailed diagnostic analysis of Calvano DGP simulation output.
Run after simulatev4.py to get a full picture of the data before R analysis.

Usage:
    python diagnose_simulation.py simulation_data_v4.csv
    python diagnose_simulation.py simulation_data_fast.csv
"""

import sys
import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────
fname = sys.argv[1] if len(sys.argv) > 1 else '/Users/louiserolland/cam_D300/algorithmic_pricing/src/simulation_data_v4.csv'
df = pd.read_csv(fname)

SEP  = "=" * 65
sep  = "-" * 65

print(f"\n{SEP}")
print(f"  SIMULATION DIAGNOSTIC REPORT")
print(f"  File : {fname}")
print(f"{SEP}")

# ─────────────────────────────────────────────────────────────
# SECTION 0 — BASIC STRUCTURE
# ─────────────────────────────────────────────────────────────
print(f"\n{'─'*65}")
print("  [0]  DATASET OVERVIEW")
print(f"{'─'*65}")
print(f"  Shape             : {df.shape[0]:,} rows × {df.shape[1]} columns")
print(f"  Columns           : {list(df.columns)}")
print(f"  Missing values    : {df.isnull().sum().sum()}")
print(f"  Unique chains     : {df['chain_id'].nunique()}")
print(f"  Sessions per cell : {df.groupby(['n','delta','mu','alpha_lr']).size().describe()[['min','mean','max']].to_dict()}")

# ─────────────────────────────────────────────────────────────
# SECTION 1 — CONVERGENCE
# ─────────────────────────────────────────────────────────────
print(f"\n{'─'*65}")
print("  [1]  CONVERGENCE ANALYSIS")
print(f"{'─'*65}")

# Overall (only Q-learning sessions can fail to converge)
ql_df = df[df['D'] == 1]
br_df = df[df['D'] == 0]

print(f"\n  Overall convergence rate (D=1 only): {ql_df['converged'].mean():.1%}  "
      f"({int(ql_df['converged'].sum())}/{len(ql_df)} Q-learning sessions)")
print(f"  Myopic sessions (D=0)              : {len(br_df)} (always converge by construction)")

print(f"\n  Convergence rate by n (Q-learning sessions only):")
conv_n = ql_df.groupby('n')['converged'].agg(
    rate='mean', n_converged='sum', n_total='count'
).reset_index()
conv_n['rate'] = conv_n['rate'].map('{:.1%}'.format)
print(conv_n.to_string(index=False))

#print(f"\n  Convergence rate by (n, delta):")
#conv_nd = ql_df.groupby(['n','delta'])['converged'].agg(
#    rate='mean', n_total='count'
#).reset_index()
#conv_nd['rate'] = conv_nd['rate'].map('{:.1%}'.format)
#print(conv_nd.to_string(index=False))

#print(f"\n  Convergence rate by (n, mu):")
#conv_nm = ql_df.groupby(['n','mu'])['converged'].agg(
#    rate='mean', n_total='count'
#).reset_index()
#conv_nm['rate'] = conv_nm['rate'].map('{:.1%}'.format)
#print(conv_nm.to_string(index=False))

#print(f"\n  Convergence rate by (n, alpha_lr):")
#conv_na = ql_df.groupby(['n','alpha_lr'])['converged'].agg(
#    rate='mean', n_total='count'
#).reset_index()
#conv_na['rate'] = conv_na['rate'].map('{:.1%}'.format)
#print(conv_na.to_string(index=False))

# ─────────────────────────────────────────────────────────────
# SECTION 2 — TREATMENT (D) AND ADOPTION
# ─────────────────────────────────────────────────────────────
print(f"\n{'─'*65}")
print("  [2]  TREATMENT ASSIGNMENT & ADOPTION")
print(f"{'─'*65}")

print(f"\n  Overall adoption rate (D=1): {df['D'].mean():.1%}  "
      f"({int(df['D'].sum())}/{len(df)})")

print(f"\n  Adoption rate by n:")
print(df.groupby('n')['D'].mean().apply(lambda x: f'{x:.1%}').rename('adoption_rate').to_string())

#print(f"\n  Adoption rate by (n, delta, mu)  [confounding structure]:")
#adopt_tab = df.groupby(['n','delta','mu'])['D'].mean().round(3)
#print(adopt_tab.to_string())

#print(f"\n  Adoption rate by alpha_lr:")
#print(df.groupby('alpha_lr')['D'].mean().apply(lambda x: f'{x:.1%}').rename('adoption_rate').to_string())

# ─────────────────────────────────────────────────────────────
# SECTION 3 — OUTCOME (DELTA)
# ─────────────────────────────────────────────────────────────
#print(f"\n{'─'*65}")
#print("  [3]  OUTCOME VARIABLE (Delta = collusion index)")
#print(f"{'─'*65}")

#print(f"\n  Full sample summary (all observations):")
#print(df['Delta'].describe().round(3).to_string())

#print(f"\n  Mean Delta by (n, D):")
#print(df.groupby(['n','D'])['Delta'].mean().round(3).to_string())

#print(f"\n  Mean Delta by (n, D) — converged Q-learning only:")
#conv_only = df[(df['D'] == 0) | ((df['D'] == 1) & df['converged'])]
#print(conv_only.groupby(['n','D'])['Delta'].mean().round(3).to_string())

#print(f"\n  Mean Delta by (n, delta) for Q-learning (D=1):")
#print(ql_df.groupby(['n','delta'])['Delta'].mean().round(3).to_string())

#print(f"\n  Mean Delta by (n, mu) for Q-learning (D=1):")
#print(ql_df.groupby(['n','mu'])['Delta'].mean().round(3).to_string())


# ─────────────────────────────────────────────────────────────
# SECTION 4 — NAIVE ATE (GROUND TRUTH)
# ─────────────────────────────────────────────────────────────
print(f"\n{'─'*65}")
print("  [4]  NAIVE AND ORACLE ATE ESTIMATES")
print(f"{'─'*65}")

naive_ate = df[df['D']==1]['Delta'].mean() - df[df['D']==0]['Delta'].mean()
print(f"\n  Naive ATE (E[Y|D=1] - E[Y|D=0])         : {naive_ate:.4f}")

# Within-cell ATE (removes confounding by covariate cell)
df['cell'] = df['n'].astype(str) + '_' + df['delta'].round(2).astype(str) + '_' + \
             df['mu'].round(2).astype(str) + '_' + df['alpha_lr'].round(2).astype(str)
cell_ates = []
for cell, grp in df.groupby('cell'):
    if grp['D'].nunique() == 2:
        ate = grp[grp['D']==1]['Delta'].mean() - grp[grp['D']==0]['Delta'].mean()
        cell_ates.append(ate)
oracle_ate = np.mean(cell_ates) if cell_ates else np.nan
print(f"  Within-cell ATE (oracle, controls for X) : {oracle_ate:.4f}  "
      f"(from {len(cell_ates)} cells with both D=0 and D=1)")

# Converged only
conv_d1 = df[(df['D']==1) & df['converged']]['Delta'].mean()
conv_d0 = df[df['D']==0]['Delta'].mean()
print(f"  Converged-only ATE (D=1 converged vs D=0): {conv_d1 - conv_d0:.4f}")

print(f"\n  Note: Calibration target from Calvano (2020):")
print(f"    n=2: Delta ~ 0.85 at (delta=0.95, mu=0.25, alpha=0.15)")
print(f"    n=3: Delta lower — Calvano reports weaker collusion with more firms")

# ─────────────────────────────────────────────────────────────
# SECTION 5 — INSTRUMENT VALIDITY
# ─────────────────────────────────────────────────────────────
print(f"\n{'─'*65}")
print("  [5]  INSTRUMENT VALIDITY (Z_hq)")
print(f"{'─'*65}")

corr_ZD = df['Z_hq'].corr(df['D'])
corr_ZY = df['Z_hq'].corr(df['Delta'])

print(f"\n  corr(Z_hq, D)     : {corr_ZD:.3f}  (relevance; target > 0.15)")
print(f"  corr(Z_hq, Delta) : {corr_ZY:.3f}  (should be small if exclusion holds)")

print(f"\n  Z_hq summary statistics:")
print(df['Z_hq'].describe().round(3).to_string())

print(f"\n  corr(Z_hq, D) by n:")
for n_val, g in df.groupby('n'):
    print(f"    n={n_val}: {g['Z_hq'].corr(g['D']):.3f}")

print(f"\n  Partial first stage (Z_hq ~ D | controlling for n, delta, mu, alpha_lr):")
try:
    from sklearn.linear_model import LinearRegression
    X_ctrl = pd.get_dummies(df[['n','delta','mu','alpha_lr']], drop_first=True).astype(float)
    Z_res = df['Z_hq'].values - LinearRegression().fit(X_ctrl, df['Z_hq']).predict(X_ctrl)
    D_res = df['D'].values    - LinearRegression().fit(X_ctrl, df['D']).predict(X_ctrl)
    partial_F = (np.corrcoef(Z_res, D_res)[0,1]**2 * (len(df)-X_ctrl.shape[1]-2)) / \
                (1 - np.corrcoef(Z_res, D_res)[0,1]**2)
    print(f"    Partial F-stat (approx): {partial_F:.1f}  (rule-of-thumb > 10 for strong IV)")
except ImportError:
    print("    (Install scikit-learn for partial F-stat calculation)")

print(f"\n  Mean Z_hq by chain_id (should vary across chains):")
chain_z = df.groupby('chain_id')['Z_hq'].mean().round(3)
print(f"    {chain_z.values.tolist()}")
print(f"    Std across chains: {chain_z.std():.3f}  (higher = better chain variation)")

# ─────────────────────────────────────────────────────────────
# SECTION 6 — COVARIATE BALANCE
# ─────────────────────────────────────────────────────────────
print(f"\n{'─'*65}")
print("  [6]  COVARIATE SUMMARY & CONFOUNDING STRENGTH")
print(f"{'─'*65}")

for col in ['n','delta','mu','alpha_lr']:
    d1_mean = df[df['D']==1][col].mean()
    d0_mean = df[df['D']==0][col].mean()
    pooled_sd = df[col].std()
    smd = (d1_mean - d0_mean) / pooled_sd if pooled_sd > 0 else 0
    print(f"  {col:10s}  D=1 mean={d1_mean:.3f}  D=0 mean={d0_mean:.3f}  "
          f"SMD={smd:+.3f}  {'⚠ strong confound' if abs(smd) > 0.3 else ''}")

print(f"\n  Jittered covariate ranges (post-jitter, as seen by R):")
for col in ['delta','mu','alpha_lr']:
    print(f"  {col:10s}  [{df[col].min():.3f}, {df[col].max():.3f}]  "
          f"(unique values: {df[col].nunique()})")

# ─────────────────────────────────────────────────────────────
# SECTION 7 — CHAIN STRUCTURE
# ─────────────────────────────────────────────────────────────
print(f"\n{'─'*65}")
print("  [7]  CHAIN STRUCTURE (for IV forest leaf variation)")
print(f"{'─'*65}")

chain_stats = df.groupby('chain_id').agg(
    n_obs    = ('D', 'count'),
    adopt_rate = ('D', 'mean'),
    mean_Z   = ('Z_hq', 'mean'),
    n_vals   = ('n', lambda x: sorted(x.unique())),
    delta_range = ('delta', lambda x: f"[{x.min():.2f},{x.max():.2f}]"),
).round(3)
print(chain_stats.to_string())

print(f"\n  Chain adoption rate std: {df.groupby('chain_id')['D'].mean().std():.3f}  "
      f"(higher = better between-chain variation in Z)")

# ─────────────────────────────────────────────────────────────
# SECTION 8 — DATA QUALITY FLAGS
# ─────────────────────────────────────────────────────────────
print(f"\n{'─'*65}")
print("  [8]  DATA QUALITY FLAGS")
print(f"{'─'*65}")

flags = []

if ql_df['converged'].mean() < 0.5:
    flags.append(f"  ⚠  Low convergence: {ql_df['converged'].mean():.1%} of Q-learning sessions. "
                 f"Consider increasing n_iter_ql or reporting converged-only results.")

if corr_ZD < 0.15:
    flags.append(f"  ⚠  Weak instrument: corr(Z,D)={corr_ZD:.3f} < 0.15. "
                 f"IV forest and 2SLS estimates will be unreliable.")

if df['D'].mean() < 0.2 or df['D'].mean() > 0.8:
    flags.append(f"  ⚠  Extreme adoption rate: {df['D'].mean():.1%}. "
                 f"GRF needs overlap (P(D=1|X) bounded away from 0 and 1).")

if df.groupby('cell').size().min() < 2:
    flags.append(f"  ⚠  Some covariate cells have < 2 observations. "
                 f"GRF leaf estimates may be noisy.")

delta_n2 = df[(df['D']==1) & (df['n']==2) & df['converged']]['Delta'].mean()
if not np.isnan(delta_n2) and (delta_n2 < 0.5 or delta_n2 > 1.0):
    flags.append(f"  ⚠  n=2 converged Delta={delta_n2:.3f}. "
                 f"Calvano target is ~0.85. Check parameter calibration.")

if not flags:
    print("\n  ✓  No major issues detected. Data looks well-formed for R analysis.")
else:
    print()
    for f in flags:
        print(f)

print(f"\n{SEP}")
print("  END OF DIAGNOSTIC REPORT")
print(f"{SEP}\n")

"""
print_terminal_output.py
Replays the post-simulation terminal output that would have appeared
after "Starting simulation..." — reads from the saved CSV.
"""

import pandas as pd
import sys
import os

# ── locate the CSV ────────────────────────────────────────────────────────────
candidates = [
    'simulation_data_v4.csv',
    os.path.join(os.path.dirname(__file__), 'simulation_data_v4.csv'),
]
# allow override via CLI:  python print_terminal_output.py path/to/file.csv
if len(sys.argv) > 1:
    candidates.insert(0, sys.argv[1])

df = None
for path in candidates:
    if os.path.exists(path):
        df = pd.read_csv(path)
        print(f"  (loaded: {path})\n")
        break

if df is None:
    print("ERROR: simulation_data_v4.csv not found.")
    print("Pass the path as an argument:  python print_terminal_output.py <path>")
    sys.exit(1)

# ── replicate run_pipeline prints ────────────────────────────────────────────

# elapsed time is not recoverable; use a placeholder
print(f"\n  Simulation complete in [see log]")

print("  Computing HQ adoption instrument...")

print(f"\n  Dataset shape   : {df.shape}")
print(f"  Convergence rate: {df['converged'].mean():.1%}")

print("\n  Mean Delta by (n, D):")
print(df.groupby(['n', 'D'])['Delta'].mean().round(3).to_string())

print("\n  Adoption rates by n:")
print(df.groupby('n')['D'].mean().round(3).to_string())

print("\n  Instrument validity — corr(Z_hq, D):")
print(f"  {df['Z_hq'].corr(df['D']):.3f}  (should be > 0.15 with random chains)")

# ── extra sanity checks (from diagnostic) ────────────────────────────────────
print("\n  Additional checks:")
print(f"  corr(Z_hq, Delta) : {df['Z_hq'].corr(df['Delta']):.3f}  "
      f"(unconditional; should be near 0 after conditioning on X)")
print(f"  Unique chains     : {df['chain_id'].nunique()}")
print(f"  Overall adoption  : {df['D'].mean():.1%}")

print(f"\n  Dataset saved to: simulation_data_v4.csv")
print("  Columns:", list(df.columns))
print("\nDone. Open analysisv5.R to run the causal inference analysis.")