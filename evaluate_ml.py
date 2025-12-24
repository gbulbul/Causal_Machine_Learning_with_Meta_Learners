
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from simulate_data import simulate_causal_data
from meta_learners import s_learner_rf, t_learner_rf
from evaluate import rmse, mae


# -----------------------------
# Simulate data
# -----------------------------
df = simulate_causal_data(n=2000, seed=42)
features = [c for c in df.columns if c.startswith("X")]

# -----------------------------
# Fit RF-based learners
# -----------------------------
tau_s_rf = s_learner_rf(df, features)
tau_t_rf = t_learner_rf(df, features)

# -----------------------------
# Errors
# -----------------------------
err_s = np.abs(df["true_tau"] - tau_s_rf)
err_t = np.abs(df["true_tau"] - tau_t_rf)

# -----------------------------
# Plot
# -----------------------------
sns.set(style="whitegrid", context="paper")
plt.figure(figsize=(6, 4))

sns.boxplot(
    data=[err_s, err_t],
    palette=["#4C72B0", "#DD8452"]
)

plt.xticks([0, 1], ["S-learner (RF)", "T-learner (RF)"])
plt.ylabel(r"Absolute Error $|\hat{\tau} - \tau|$")
plt.title("Random Forest Meta-Learners\nNonlinear & Heterogeneous Treatment Effects")

plt.tight_layout()
plt.show()

# -----------------------------
# Numeric summary
# -----------------------------
print("RF-based evaluation")
print("S-learner RF RMSE:", rmse(df["true_tau"], tau_s_rf))
print("T-learner RF RMSE:", rmse(df["true_tau"], tau_t_rf))
print("S-learner RF MAE :", mae(df["true_tau"], tau_s_rf))
print("T-learner RF MAE :", mae(df["true_tau"], tau_t_rf))
