# =============================================================
# D300 Causal Inference and Machine Learning — Analysis Script
# =============================================================
# Uses simulation_data.csv produced by simulate.py
#
# Estimators:
#   1. Naive OLS          (confounded baseline)
#   2. OLS + controls     (partial deconfounding)
#   3. OLS + interactions (parametric benchmark with pre-specified HTE)
#   4. IV regression      (instruments D with Z_hq — mirrors Assad et al.)
#   5. Post-Double-Selection LASSO  (high-dim variable selection)
#   6. Causal Forest (primary HTE) + IV Forest (ATE robustness)
#
# Output: cost-benefit analysis comparing parametric vs ML methods
# =============================================================

# Install packages if needed (run once)
# install.packages(c("grf", "hdm", "ivreg", "ggplot2", "dplyr", "fixest", "patchwork", "sandwich", "lmtest"))

library(grf)
library(hdm)          # Post-Double-Selection LASSO
library(ivreg)        # IV regression
library(ggplot2)
library(dplyr)
library(tidyr)
library(scales)
library(patchwork)    # combine ggplots
library(fixest)       # fast fixed-effects / IV
library(sandwich)     # HC standard errors for DML
library(lmtest)       # coeftest for robust inference

set.seed(42)

# =============================================================
# 0. LOAD AND PREPARE DATA
# =============================================================

# Ensure working directory is the script's own folder
setwd(dirname(rstudioapi::getSourceEditorContext()$path))

# Change path if using fast mode output
df <- read.csv("../data/simulation_data_v2.csv")

cat("Dataset dimensions:", nrow(df), "x", ncol(df), "\n")
cat("Columns:", paste(names(df), collapse=", "), "\n\n")

# Core variables
Y <- df$Delta         # outcome: profit gain (0 = Nash, 1 = monopoly)
D <- df$D             # treatment: 1 = both firms algorithmic, 0 = human
Z <- df$Z_hq          # instrument: HQ adoption rate in same chain

# Covariates (potential confounders / effect modifiers)
X_raw <- df[, c("n", "delta", "mu", "alpha_lr")]
X     <- as.matrix(X_raw)

# Quick descriptive check
cat("--- Descriptive Statistics ---\n")
cat("Mean Delta by (n, D):\n")
print(df %>% group_by(n, D) %>% summarise(
  mean_Delta = mean(Delta), sd_Delta = sd(Delta), N = n(), .groups = "drop"
))

cat("\nAdoption rates by n:\n")
print(df %>% group_by(n) %>% summarise(adoption_rate = mean(D), .groups = "drop"))

cat("\nCorr(Z_hq, D):", round(cor(Z, D), 3), " [instrument relevance]\n\n")

# =============================================================
# 1. NAIVE OLS  (confounded)
# =============================================================

ols_naive <- lm(Delta ~ D, data = df)
cat("=== 1. NAIVE OLS (no controls) ===\n")
print(summary(ols_naive)$coefficients["D", ])

# =============================================================
# 2. OLS WITH CONTROLS  (but no interaction terms)
# =============================================================

ols_controls <- lm(Delta ~ D + n + delta + mu + alpha_lr, data = df)
cat("\n=== 2. OLS + CONTROLS (no interactions) ===\n")
print(summary(ols_controls)$coefficients["D", ])

# =============================================================
# 3. OLS WITH PRE-SPECIFIED INTERACTIONS  (parametric benchmark)
# Theory: delta and n are the key moderators (patience + concentration)
# =============================================================

ols_interact <- lm(
  Delta ~ D + n + delta + mu + alpha_lr +
    D:n + D:delta + D:mu,
  data = df
)
cat("\n=== 3. OLS + INTERACTIONS (parametric benchmark) ===\n")
summary_interact <- summary(ols_interact)$coefficients
print(summary_interact)

# Implied CATE from parametric model:
# tau_ols(x) = beta_D + gamma_n * n + gamma_delta * delta + gamma_mu * mu
tau_ols <- predict(ols_interact) - predict(
  lm(Delta ~ n + delta + mu + alpha_lr, data = df)
)
# More direct: use model matrix
coef_int <- coef(ols_interact)
tau_ols_manual <- coef_int["D"] +
  coef_int["D:n"]     * df$n     +
  coef_int["D:delta"] * df$delta +
  coef_int["D:mu"]    * df$mu
df$tau_ols <- tau_ols_manual

# =============================================================
# 4. IV REGRESSION  (HQ adoption as instrument)
# Mirrors Assad et al.'s identification strategy
# =============================================================

iv_model <- ivreg(
  Delta ~ D + n + delta + mu + alpha_lr | Z_hq + n + delta + mu + alpha_lr,
  data = df
)
cat("\n=== 4. IV REGRESSION (parametric benchmark) ===\n")
iv_summary <- summary(iv_model, diagnostics = TRUE)
print(iv_summary$coefficients["D", ])
cat("First-stage F-statistic:", round(iv_summary$diagnostics["Weak instruments", "statistic"], 2), "\n")

# =============================================================
# 5. POST-DOUBLE-SELECTION LASSO  (high-dim variable selection)
# Builds an expanded covariate space including all interactions
# then uses double-selection to choose controls
# =============================================================

cat("\n=== 5. POST-DOUBLE-SELECTION LASSO ===\n")

# Expand covariate matrix: add all pairwise interactions and squares
X_expand <- model.matrix(
  ~ (n + delta + mu + alpha_lr)^2 + I(delta^2) + I(mu^2),
  data = df
)[, -1]  # drop intercept

# PDS-LASSO via hdm
pds_result <- rlassoEffect(
  x = X_expand,
  y = Y,
  d = D,
  method = "partialling out"
)
cat("PDS-LASSO ATE estimate:\n")
print(summary(pds_result))

# Which controls did LASSO select?
cat("\nLASSO selected controls (outcome equation):\n")
lasso_y <- rlasso(X_expand, Y)
selected_y <- names(lasso_y$coefficients[lasso_y$coefficients != 0])
cat(paste(selected_y, collapse = ", "), "\n")

cat("\nLASSO selected controls (treatment equation):\n")
lasso_d <- rlasso(X_expand, D)
selected_d <- names(lasso_d$coefficients[lasso_d$coefficients != 0])
cat(paste(selected_d, collapse = ", "), "\n")

# =============================================================
# 5b. DOUBLE MACHINE LEARNING  (explicit cross-fitted ATE)
#
# Manual K-fold cross-fitting with rlasso nuisance estimators.
# Implements the Neyman-orthogonal PLR moment condition:
#   E[(Y - l(X))(D - m(X))] = tau * E[(D - m(X))^2]
# where l(x) = E[Y|X], m(x) = E[D|X] are estimated by Post-LASSO.
# Cross-fitting removes first-stage regularisation bias from tau.
# HC3 standard errors account for heteroskedasticity in residuals.
# =============================================================

cat("\n=== 5b. DOUBLE ML (manual cross-fitting, K=5) ===\n")

K       <- 5
set.seed(42)
folds   <- sample(rep(1:K, length.out = nrow(df)))
Y_tilde <- numeric(nrow(df))
D_tilde <- numeric(nrow(df))

for (k in 1:K) {
  train_idx <- which(folds != k)
  test_idx  <- which(folds == k)

  X_tr <- X_expand[train_idx, ]
  X_te <- X_expand[test_idx,  ]

  # Nuisance l(x) = E[Y|X] — Post-LASSO on training fold
  lasso_y_k         <- rlasso(X_tr, Y[train_idx], post = TRUE)
  Y_tilde[test_idx] <- Y[test_idx] - predict(lasso_y_k, X_te)

  # Nuisance m(x) = E[D|X] — Post-LASSO on training fold
  lasso_d_k         <- rlasso(X_tr, D[train_idx], post = TRUE)
  D_tilde[test_idx] <- D[test_idx] - predict(lasso_d_k, X_te)
}

# DML estimator: Frisch-Waugh regression of Y_tilde on D_tilde
dml_fit  <- lm(Y_tilde ~ D_tilde)
tau_dml  <- coef(dml_fit)["D_tilde"]
# HC3 robust SEs — valid under heteroskedastic residuals
se_dml   <- sqrt(vcovHC(dml_fit, type = "HC3")["D_tilde", "D_tilde"])

cat(sprintf("DML ATE:  %.4f\n", tau_dml))
cat(sprintf("DML SE (HC3): %.4f\n", se_dml))
cat(sprintf("DML 95%% CI: [%.4f, %.4f]\n",
            tau_dml - 1.96 * se_dml, tau_dml + 1.96 * se_dml))

# Store residuals for diagnostics
df$Y_tilde <- Y_tilde
df$D_tilde <- D_tilde

# First-stage R² of nuisance models (sanity check)
cat(sprintf("\nNuisance fit check (fold-averaged):\n"))
cat(sprintf("  Var(Y_tilde)/Var(Y) = %.3f  [fraction of Y unexplained by X]\n",
            var(Y_tilde) / var(Y)))
cat(sprintf("  Var(D_tilde)/Var(D) = %.3f  [fraction of D unexplained by X]\n",
            var(D_tilde) / var(D)))

# =============================================================
# 6. CAUSAL FOREST  (GRF — main HTE estimator)
#    + IV Forest ATE robustness check
#
# Since X is fully observed by construction in the simulation,
# conditional unconfoundedness holds exactly. Causal forest is
# therefore the correct primary estimator for CATEs.
# The IV forest is retained only as an ATE robustness check;
# it should NOT be used for individual CATEs here because the
# within-leaf first stage is too weak after conditioning on X.
# =============================================================

cat("\n=== 6. GRF CAUSAL FOREST (primary) + IV robustness ===\n")

# --- 6a. Causal forest (primary) ---
cf <- causal_forest(
  X = X,
  Y = Y,
  W = D,
  num.trees       = 2000,
  honesty         = TRUE,
  tune.parameters = "all",
  seed            = 42
)

ate_cf <- average_treatment_effect(cf)
cat("Causal forest ATE:", round(ate_cf[1], 3),
    " SE:", round(ate_cf[2], 3), "\n")

# CATEs — causal forest predictions are stable (no IV denominator blowup)
tau_grf    <- predict(cf)$predictions
df$tau_grf <- tau_grf

# Variable importance
vim <- variable_importance(cf)
names(vim) <- colnames(X)
cat("\nVariable importance for treatment effect heterogeneity:\n")
print(sort(vim, decreasing = TRUE))

# Best linear projection — tests genuine heterogeneity
cat("\nBest Linear Projection (test for genuine heterogeneity):\n")
blp <- best_linear_projection(cf, X)
print(blp)

# --- 6c. GATES — Group Average Treatment Effects ---
# Partition observations into quintiles of predicted CATE,
# then estimate the ATE within each group using the fitted forest.
# A monotone pattern (Q1 < Q2 < ... < Q5) confirms that the CATE
# ranking is informative, not noise.
cat("\n--- GATES: Group Average Treatment Effects (CATE quintiles) ---\n")

tau_breaks  <- quantile(tau_grf, probs = seq(0, 1, by = 0.2), na.rm = TRUE)
df$cate_grp <- as.integer(cut(tau_grf, breaks = tau_breaks,
                               include.lowest = TRUE, labels = FALSE))

gates_list <- lapply(1:5, function(g) {
  idx   <- which(df$cate_grp == g)
  ate_g <- average_treatment_effect(cf, subset = idx)
  data.frame(
    CATE_Quintile = g,
    GATE          = round(ate_g[1], 4),
    SE            = round(ate_g[2], 4),
    CI_lo         = round(ate_g[1] - 1.96 * ate_g[2], 4),
    CI_hi         = round(ate_g[1] + 1.96 * ate_g[2], 4),
    N             = length(idx),
    Mean_tau_pred = round(mean(tau_grf[idx]), 4)
  )
})
gates_df <- bind_rows(gates_list)
cat("\nGATES (Q1 = lowest predicted collusion gain, Q5 = highest):\n")
print(gates_df)

# Monotonicity test: Spearman rank correlation across GATE quintiles
gates_spearman <- cor(gates_df$CATE_Quintile, gates_df$GATE,
                      method = "spearman")
cat(sprintf("\nSpearman rank correlation (quintile vs GATE): %.3f\n",
            gates_spearman))
cat("(+1 = perfectly monotone CATE ranking — confirms forest discriminates well)\n")

# --- 6d. CLAN — Classification Analysis ---
# Compare covariate means between the top and bottom CATE quintile.
# Significant differences identify which market characteristics predict
# who benefits most from algorithmic pricing adoption.
cat("\n--- CLAN: Classification Analysis (top vs bottom CATE quintile) ---\n")

top_idx <- which(df$cate_grp == 5)
bot_idx <- which(df$cate_grp == 1)

clan_list <- lapply(colnames(X_raw), function(v) {
  tt <- t.test(df[[v]][top_idx], df[[v]][bot_idx])
  data.frame(
    Variable     = v,
    Mean_Top_Q5  = round(mean(df[[v]][top_idx]), 4),
    Mean_Bot_Q1  = round(mean(df[[v]][bot_idx]), 4),
    Difference   = round(mean(df[[v]][top_idx]) - mean(df[[v]][bot_idx]), 4),
    p_value      = round(tt$p.value, 4),
    Signif       = ifelse(tt$p.value < 0.01, "***",
                   ifelse(tt$p.value < 0.05, "**",
                   ifelse(tt$p.value < 0.10, "*", "")))
  )
})
clan_df <- bind_rows(clan_list)
cat("\nCovariate means: top quintile (Q5) vs bottom quintile (Q1) of CATE\n")
cat("(Q5 = markets with largest predicted algorithmic collusion gain)\n\n")
print(clan_df)
cat("\nInterpretation guide: positive Difference means higher covariate value\n")
cat("predicts a larger collusion gain from algorithmic adoption.\n")

# --- 6b. IV forest — ATE robustness only ---
# Used only to produce an IV-based ATE for Table 1.
# CATEs from this forest are discarded (within-leaf instrument too weak).
cat("\n--- IV Forest (ATE robustness check only) ---\n")
cat("Z_hq summary:\n"); print(summary(Z))
Z_mat <- matrix(as.integer(Z > median(Z)), ncol = 1)
cat("Z_binary distribution:\n"); print(table(Z_mat[,1]))

fs_check  <- lm(D ~ Z_mat + n + delta + mu + alpha_lr, data = df)
fs_f_stat <- summary(fs_check)$fstatistic[1]
cat(sprintf("First-stage F (Z on D | X): %.2f", fs_f_stat))
if (fs_f_stat < 10) {
  cat("  *** WARNING: weak instrument ***\n")
} else {
  cat("  [instrument relevant]\n")
}

iv_forest <- instrumental_forest(
  X = X, Y = Y, W = D, Z = Z_mat,
  num.trees = 2000, honesty = TRUE,
  tune.parameters = "all", seed = 42
)
ate_grf <- average_treatment_effect(iv_forest)
cat("IV forest ATE:", round(ate_grf[1], 3),
    " SE:", round(ate_grf[2], 3),
    " (robustness only — see note above)\n")

# =============================================================
# 7. COST-BENEFIT ANALYSIS
# =============================================================

cat("\n=== 7. COST-BENEFIT ANALYSIS ===\n")

# --- 7a. ATE comparison across methods ---
# Causal forest is the primary ML estimator.
# IV forest ATE included as robustness; its CATEs are unreliable (weak within-leaf F).
ate_table <- data.frame(
  Method = c("Naive OLS", "OLS + controls", "OLS + interactions",
             "IV (parametric)", "PDS-LASSO", "DML (cross-fitted)",
             "Causal Forest", "IV Forest (robust.)"),
  ATE_estimate = c(
    coef(ols_naive)["D"],
    coef(ols_controls)["D"],
    coef(ols_interact)["D"],
    coef(iv_model)["D"],
    coef(pds_result)[1],
    tau_dml,
    ate_cf[1],    # causal forest — primary ML estimate
    ate_grf[1]    # IV forest — robustness only
  ),
  SE = c(
    summary(ols_naive)$coefficients["D", "Std. Error"],
    summary(ols_controls)$coefficients["D", "Std. Error"],
    summary(ols_interact)$coefficients["D", "Std. Error"],
    summary(iv_model)$coefficients["D", "Std. Error"],
    summary(pds_result)$coefficients[1, "Std. Error"],
    se_dml,
    ate_cf[2],
    ate_grf[2]
  )
)
cat("\nATE Summary Table:\n")
ate_table[, -1] <- round(ate_table[, -1], 4)
print(ate_table)

# --- 7b. Heterogeneity comparison: OLS interaction vs GRF ---
# Bin by n and delta to compare
df$tau_grf   <- tau_grf
df$tau_ols   <- tau_ols_manual

cate_comparison <- df %>%
  filter(D == 1) %>%        # only treated for cleaner comparison
  group_by(n, delta = round(delta, 2)) %>%
  summarise(
    tau_grf_mean = mean(tau_grf),
    tau_ols_mean = mean(tau_ols),
    N = n(),
    .groups = "drop"
  )
cat("\nCate comparison by (n, delta):\n")
print(cate_comparison)

# RMSE: how much do the two methods disagree on CATE?
rmse_ols_vs_grf <- sqrt(mean((df$tau_ols - df$tau_grf)^2))
cat("\nRMSE(OLS interactions vs GRF CATE):", round(rmse_ols_vs_grf, 4), "\n")

# --- 7c. Finite-sample variance analysis (core cost of ML) ---
# We use causal_forest (not instrumental_forest) here because IV-based forests
# are unreliable at N < 500: first-stage noise dominates. The causal forest
# is valid under conditional unconfoundedness — a reasonable benchmark for
# illustrating the variance cost of ML vs OLS.
cat("\n--- Finite-sample variance analysis ---\n")
cat("(Causal forest used for stability; instrumental forest needs N >= 500)\n")
sample_sizes <- c(100, 250, 500, 1000, min(2000, nrow(df)))

fs_results <- lapply(sample_sizes, function(n_sub) {
  if (n_sub >= nrow(df)) {
    idx <- 1:nrow(df)
  } else {
    idx <- sample(1:nrow(df), n_sub, replace = FALSE)
  }

  sub_X  <- X[idx, ]
  sub_Y  <- Y[idx]
  sub_D  <- D[idx]
  sub_df <- df[idx, ]

  # OLS with interactions at this sample size
  ols_sub    <- lm(Delta ~ D + n + delta + mu + alpha_lr + D:n + D:delta + D:mu,
                   data = sub_df)
  ate_ols_sub <- coef(ols_sub)["D"]
  se_ols_sub  <- summary(ols_sub)$coefficients["D", "Std. Error"]

  # Causal forest (stable across sample sizes; conditional unconfoundedness)
  tryCatch({
    cf_sub  <- causal_forest(
      sub_X, sub_Y, sub_D,
      num.trees = 500, honesty = TRUE, seed = 42
    )
    ate_sub <- average_treatment_effect(cf_sub)
    # Guard against NaN (can happen if D has no variation in tiny subsample)
    if (is.nan(ate_sub[1]) || is.na(ate_sub[1])) stop("NaN/NA ATE")
    list(
      n_sub    = n_sub,
      ate_grf  = ate_sub[1],
      se_grf   = ate_sub[2],
      ate_ols  = ate_ols_sub,
      se_ols   = se_ols_sub,
      cate_var = var(predict(cf_sub)$predictions)
    )
  }, error = function(e) {
    list(n_sub    = n_sub,
         ate_grf  = NA_real_,
         se_grf   = NA_real_,
         ate_ols  = ate_ols_sub,
         se_ols   = se_ols_sub,
         cate_var = NA_real_)
  })
})

fs_df <- bind_rows(lapply(fs_results, as.data.frame))
cat("\nFinite-sample results:\n")
print(round(fs_df, 4))

# =============================================================
# 8. PLOTS
# =============================================================

cat("\n=== 8. GENERATING PLOTS ===\n")

# Shared theme and palette
clr_grf  <- "#3B82F6"   # blue
clr_ols  <- "#F97316"   # orange
clr_n2   <- "#6366F1"   # indigo  (duopoly)
clr_n3   <- "#EC4899"   # pink    (triopoly)

theme_clean <- function(base_size = 12) {
  theme_minimal(base_size = base_size) %+replace%
    theme(
      plot.title       = element_text(face = "bold", size = base_size + 1,
                                      margin = margin(b = 4)),
      plot.subtitle    = element_text(colour = "grey45", size = base_size - 1,
                                      margin = margin(b = 10)),
      plot.caption     = element_text(colour = "grey60", size = base_size - 3,
                                      hjust = 0, margin = margin(t = 8)),
      axis.title       = element_text(size = base_size - 1, colour = "grey30"),
      axis.text        = element_text(size = base_size - 2, colour = "grey40"),
      panel.grid.major = element_line(colour = "grey92", linewidth = 0.4),
      panel.grid.minor = element_blank(),
      legend.position  = "bottom",
      legend.title     = element_text(size = base_size - 2, face = "bold"),
      legend.text      = element_text(size = base_size - 2),
      strip.text       = element_text(face = "bold", size = base_size - 1),
      plot.margin      = margin(12, 14, 8, 12)
    )
}

# --- Plot 1: GRF vs OLS CATE scatter, faceted by n ---
p1 <- ggplot(df, aes(x = tau_ols, y = tau_grf, colour = factor(n))) +
  geom_point(alpha = 0.25, size = 0.9, stroke = 0) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed",
              colour = "grey30", linewidth = 0.6) +
  geom_smooth(method = "lm", se = FALSE, linewidth = 1,
              aes(group = factor(n))) +
  facet_wrap(~paste0(n, " firms"), ncol = 2) +
  scale_colour_manual(values = c("2" = clr_n2, "3" = clr_n3),
                      name = "Market size") +
  labs(
    title    = "GRF vs OLS: where do the CATE estimates diverge?",
    subtitle = "Dashed line = perfect agreement. Fitted line shows systematic differences.",
    x        = "OLS interaction model  τ̂",
    y        = "Causal forest  τ̂",
    caption  = "Each point is one market observation."
  ) +
  theme_clean() +
  guides(colour = "none")   # facet labels already identify n

# --- Plot 2: CATE along delta, split by n ---
# Use 6 quantile bins instead of round(delta,2) — jitter means rounding
# produces 50+ micro-groups with 2-5 obs each, making lines jagged.
# Quantile bins give ~180 obs per group and smooth, readable curves.
df <- df %>%
  mutate(delta_bin = cut(delta,
                         breaks = quantile(delta, probs = seq(0, 1, by = 1/6),
                                           na.rm = TRUE),
                         include.lowest = TRUE, labels = FALSE))

df_plot2 <- df %>%
  group_by(n, delta_bin) %>%
  summarise(
    delta_mid    = mean(delta),
    tau_grf_mean = mean(tau_grf),
    tau_grf_lo   = mean(tau_grf) - 1.96 * sd(tau_grf) / sqrt(n()),
    tau_grf_hi   = mean(tau_grf) + 1.96 * sd(tau_grf) / sqrt(n()),
    tau_ols_mean = mean(tau_ols),
    .groups = "drop"
  )

p2 <- ggplot(df_plot2, aes(x = delta_mid, colour = factor(n), fill = factor(n))) +
  geom_ribbon(aes(ymin = tau_grf_lo, ymax = tau_grf_hi), alpha = 0.12,
              colour = NA) +
  geom_line(aes(y = tau_grf_mean, linetype = "GRF"), linewidth = 1.1) +
  geom_line(aes(y = tau_ols_mean, linetype = "OLS"), linewidth = 0.85) +
  geom_point(aes(y = tau_grf_mean), size = 2, shape = 21,
             fill = "white", stroke = 0.8) +
  scale_colour_manual(values = c("2" = clr_n2, "3" = clr_n3), name = "Firms") +
  scale_fill_manual(values   = c("2" = clr_n2, "3" = clr_n3), guide = "none") +
  scale_linetype_manual(values = c("GRF" = "solid", "OLS" = "dashed"),
                        name = "Method") +
  labs(
    title    = "How patience (δ) shapes the collusion gain",
    subtitle = "GRF 95% CI shaded (6 quantile bins). n=2 vs n=3 separation = market-structure heterogeneity.",
    x        = "Discount factor  δ  (bin midpoint)",
    y        = "CATE  τ̂",
    caption  = "Higher δ = more patient firms; collusion gain rises then flattens near δ ≈ 0.85."
  ) +
  theme_clean()

# --- Plot 3: SE vs sample size — cost of ML ---
fs_long <- fs_df %>%
  filter(!is.na(se_grf)) %>%
  tidyr::pivot_longer(cols = c(se_grf, se_ols),
                      names_to = "method", values_to = "se") %>%
  mutate(method = ifelse(method == "se_grf", "Causal forest", "OLS interactions"))

p3 <- ggplot(fs_long, aes(x = n_sub, y = se, colour = method)) +
  geom_line(linewidth = 1.1) +
  geom_point(size = 2.5, stroke = 0.3, fill = "white", shape = 21) +
  scale_colour_manual(values = c("Causal forest"    = clr_grf,
                                 "OLS interactions" = clr_ols),
                      name = NULL) +
  scale_x_log10(breaks = c(100, 250, 500, 1000, 2000),
                labels = scales::comma) +
  scale_y_continuous(labels = scales::number_format(accuracy = 0.001)) +
  labs(
    title    = "The cost of ML: standard error vs sample size",
    subtitle = "Causal forest needs more data to match OLS precision — the bias-variance trade-off.",
    x        = "Sample size  N  (log scale)",
    y        = "SE of ATE estimate",
    caption  = "OLS is efficient under correct specification; causal forest pays a variance tax for flexibility."
  ) +
  theme_clean()

# --- Plot 4: CATE heatmap — n × delta bins ---
# delta_bin already created above (6 quantile groups).
# Label each bin with its midpoint range for readability.
df_heat <- df %>%
  group_by(n, delta_bin) %>%
  summarise(
    tau_grf   = mean(tau_grf),
    delta_mid = mean(delta),
    .groups = "drop"
  ) %>%
  mutate(
    label     = sprintf("%.2f", tau_grf),
    delta_lab = sprintf("%.2f", delta_mid)
  )

p4 <- ggplot(df_heat,
             aes(x = factor(n), y = reorder(delta_lab, delta_mid),
                 fill = tau_grf)) +
  geom_tile(colour = "white", linewidth = 0.8) +
  geom_text(aes(label = label), size = 3.8, fontface = "bold",
            colour = ifelse(df_heat$tau_grf > 0.30, "white", "grey20")) +
  scale_fill_gradientn(
    colours = c("#FEF3C7", "#FCD34D", "#F97316", "#B45309"),
    name    = "CATE  τ̂",
    limits  = c(0, NA)
  ) +
  labs(
    title   = "Causal forest CATE by market structure and patience",
    x       = "Number of firms  n",
    y       = "Discount factor  δ",
    caption = "Warmer colour = larger algorithmic collusion gain."
  ) +
  theme_clean() +
  theme(panel.grid = element_blank(),
        legend.position = "right")

# --- Plot 5: ATE forest plot across all methods ---
ate_plot_df <- ate_table %>%
  mutate(
    lo     = ATE_estimate - 1.96 * SE,
    hi     = ATE_estimate + 1.96 * SE,
    Method = factor(Method, levels = rev(Method))
  )

p5 <- ggplot(ate_plot_df, aes(y = Method, x = ATE_estimate,
                               xmin = lo, xmax = hi,
                               colour = Method)) +
  geom_vline(xintercept = 0, linetype = "dashed", colour = "grey50",
             linewidth = 0.5) +
  geom_errorbarh(height = 0.25, linewidth = 0.9) +
  geom_point(size = 3.5) +
  scale_colour_manual(values = c(
    "Naive OLS"           = "#94A3B8",
    "OLS + controls"      = "#64748B",
    "OLS + interactions"  = clr_ols,
    "IV (parametric)"     = "#10B981",
    "PDS-LASSO"           = "#8B5CF6",
    "DML (cross-fitted)"  = "#EC4899",
    "Causal Forest"       = clr_grf,
    "IV Forest (robust.)" = "#93C5FD"
  )) +
  labs(
    title    = "ATE estimates and 95% CIs across all methods",
    subtitle = "Naive OLS is biased upward; IV and ML methods agree more closely.",
    x        = "Average Treatment Effect  τ̂",
    y        = NULL,
    caption  = "Outcome: profit gain relative to Nash (0) and monopoly (1) benchmarks."
  ) +
  theme_clean() +
  guides(colour = "none")

# --- Plot 6: GRF CATE distribution by treatment status ---
p6 <- ggplot(df, aes(x = tau_grf, fill = factor(D), colour = factor(D))) +
  geom_density(alpha = 0.25, linewidth = 0.9) +
  geom_vline(data = df %>% group_by(D) %>% summarise(m = mean(tau_grf)),
             aes(xintercept = m, colour = factor(D)),
             linetype = "dashed", linewidth = 0.8) +
  scale_fill_manual(values  = c("0" = clr_ols, "1" = clr_grf),
                    labels  = c("0" = "Human (D=0)", "1" = "Algorithmic (D=1)"),
                    name    = NULL) +
  scale_colour_manual(values = c("0" = clr_ols, "1" = clr_grf),
                      labels = c("0" = "Human (D=0)", "1" = "Algorithmic (D=1)"),
                      name   = NULL) +
  labs(
    title    = "Distribution of individual CATEs by treatment status",
    subtitle = "Dashed lines mark group means. Overlap shows where effects are similar.",
    x        = "GRF CATE  τ̂",
    y        = "Density",
    caption  = "Treated markets (D=1) do not necessarily have higher CATEs — selection is confounded."
  ) +
  theme_clean()

# --- Plot 7: GATES — ATE by CATE quintile ---
p7 <- ggplot(gates_df, aes(x = CATE_Quintile, y = GATE,
                            ymin = CI_lo, ymax = CI_hi)) +
  geom_hline(yintercept = ate_cf[1], linetype = "dashed",
             colour = "grey50", linewidth = 0.6) +
  geom_errorbar(width = 0.2, linewidth = 0.9, colour = clr_grf) +
  geom_point(size = 4, colour = clr_grf) +
  geom_line(colour = clr_grf, linewidth = 0.8, linetype = "dotted") +
  scale_x_continuous(breaks = 1:5,
                     labels = c("Q1\n(lowest)", "Q2", "Q3", "Q4",
                                "Q5\n(highest)")) +
  annotate("text", x = 4.5, y = ate_cf[1] + 0.005,
           label = sprintf("ATE = %.3f", ate_cf[1]),
           colour = "grey40", size = 3.5) +
  labs(
    title    = "GATES: Group Average Treatment Effects by CATE quintile",
    subtitle = "Monotone pattern confirms the causal forest ranks markets correctly.",
    x        = "Predicted CATE quintile (Q1 = lowest collusion gain)",
    y        = "GATE  (95% CI)",
    caption  = "Dashed line = overall ATE from causal forest."
  ) +
  theme_clean()

# --- Plot 8: CLAN — covariate differences top vs bottom quintile ---
clan_plot_df <- clan_df %>%
  mutate(
    Variable = factor(Variable, levels = Variable),
    colour   = ifelse(Difference > 0, clr_grf, clr_ols),
    label    = paste0(Signif, " (p=", sprintf("%.3f", p_value), ")")
  )

p8 <- ggplot(clan_plot_df, aes(x = Variable, y = Difference, fill = colour)) +
  geom_col(width = 0.55, colour = "white") +
  geom_text(aes(label = label,
                vjust = ifelse(Difference >= 0, -0.4, 1.4)),
            size = 3.2, colour = "grey30") +
  geom_hline(yintercept = 0, linewidth = 0.5, colour = "grey40") +
  scale_fill_identity() +
  labs(
    title    = "CLAN: who benefits most from algorithmic adoption?",
    subtitle = "Difference in covariate means: top CATE quintile minus bottom quintile.",
    x        = NULL,
    y        = "Mean difference (Q5 − Q1)",
    caption  = "Positive = higher covariate value predicts larger collusion gain. *** p<0.01."
  ) +
  theme_clean()

# --- Save ---
# Page 1: core heterogeneity story
page1 <- (p2 + p4) / (p5 + p6)
ggsave("cate_analysis.png", page1, width = 14, height = 10, dpi = 180)
cat("  Saved: cate_analysis.png\n")

# Page 2: Chernozhukov et al. heterogeneity ladder (GATES + CLAN)
page2 <- p7 + p8
ggsave("gates_clan.png", page2, width = 14, height = 5, dpi = 180)
cat("  Saved: gates_clan.png\n")

# Individual saves
ggsave("ate_comparison.png",  p5, width = 8,  height = 5,  dpi = 180)
cat("  Saved: ate_comparison.png\n")

ggsave("cate_by_delta.png",   p2, width = 8,  height = 5,  dpi = 180)
cat("  Saved: cate_by_delta.png\n")

ggsave("cate_heatmap.png",    p4, width = 6,  height = 5,  dpi = 180)
cat("  Saved: cate_heatmap.png\n")

ggsave("finite_sample.png",   p3, width = 7,  height = 5,  dpi = 180)
cat("  Saved: finite_sample.png\n")

ggsave("cate_scatter.png",    p1, width = 9,  height = 5,  dpi = 180)
cat("  Saved: cate_scatter.png\n")

ggsave("cate_density.png",    p6, width = 7,  height = 5,  dpi = 180)
cat("  Saved: cate_density.png\n")

# =============================================================
# 9. PAPER-READY SUMMARY TABLE
# =============================================================

cat("\n=== 9. PAPER-READY OUTPUT ===\n")

cat("\nTable 1: ATE Estimates Across Methods\n")
cat(sprintf("%-25s %10s %10s %10s\n", "Method", "ATE", "SE", "95% CI"))
cat(strrep("-", 58), "\n")
for (i in 1:nrow(ate_table)) {
  lo <- ate_table$ATE_estimate[i] - 1.96 * ate_table$SE[i]
  hi <- ate_table$ATE_estimate[i] + 1.96 * ate_table$SE[i]
  cat(sprintf("%-25s %10.4f %10.4f [%6.4f, %6.4f]\n",
      ate_table$Method[i], ate_table$ATE_estimate[i],
      ate_table$SE[i], lo, hi))
}

cat("\nTable 2: GRF Variable Importance (treatment effect heterogeneity)\n")
vim_df <- data.frame(Variable = names(vim), Importance = as.numeric(vim))
vim_df <- vim_df[order(-vim_df$Importance), ]
print(vim_df)

cat("\nTable 3: GATES — Group Average Treatment Effects by CATE quintile\n")
print(gates_df)

cat("\nTable 4: CLAN — Covariate means by CATE extremes (Q5 vs Q1)\n")
print(clan_df)

cat("\n--- Analysis complete ---\n")
cat("Key outputs: cate_analysis.png, gates_clan.png, finite_sample.png\n")
