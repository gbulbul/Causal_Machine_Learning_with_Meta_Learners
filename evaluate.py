import numpy as np
from simulate_data import simulate_causal_data
from meta_learners import s_learner, t_learner
import matplotlib.pyplot as plt

def rmse(x, y):
    return np.sqrt(np.mean((x - y) ** 2))

if __name__ == "__main__":

    # simulate data
    df = simulate_causal_data()
    features = [c for c in df.columns if c.startswith("X")]

    tau_s = s_learner(df, features)
    tau_t = t_learner(df, features)

    # metrics
    rmse_s = rmse(df["true_tau"], tau_s)
    rmse_t = rmse(df["true_tau"], tau_t)

    # obtain figure
    plt.figure(figsize=(6,4))
    plt.bar(["S-learner", "T-learner"], [rmse_s, rmse_t])
    plt.ylabel("RMSE")
    plt.title("Meta-Learner Comparison")
    plt.tight_layout()
    plt.savefig("figures/figure_1_meta_learner_rmse.png", dpi=300)
    plt.close()

