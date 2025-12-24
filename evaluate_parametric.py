import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

from simulate_data import simulate_causal_data
from meta_learners import s_learner, t_learner
from evaluate import rmse, mae

if __name__ == "__main__":

    # Paths
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    FIG_DIR = os.path.join(BASE_DIR, "figures")
    os.makedirs(FIG_DIR, exist_ok=True)

    # Simulate heterogenous data
    df = simulate_causal_data(n=2000)
    features = [c for c in df.columns if c.startswith("X")]

    # Meta-learners
    tau_s = s_learner(df, features)
    tau_t = t_learner(df, features)

    # Individual-level absolute errors
    err_s = np.abs(df["true_tau"] - tau_s)
    err_t = np.abs(df["true_tau"] - tau_t)

    # Plot
    sns.set(style="whitegrid", context="paper")

    plt.figure(figsize=(6, 4))
    sns.boxplot(
        data=[err_s, err_t],
        palette=["#4C72B0", "#DD8452"]
    )

    plt.xticks([0, 1], ["S-learner", "T-learner"])
    plt.ylabel(r"Absolute Error $|\hat{\tau} - \tau|$")
    plt.title("Meta-Learner Performance under Strong Heterogeneity")

    plt.tight_layout()
    plt.savefig(
        os.path.join(FIG_DIR, "figure_2_meta_learner_heterogeneity.png"),
        dpi=300
    )
    plt.close()

    print("Evaluate 2 results (heterogeneous τ):")
    print("S-learner RMSE:", rmse(df["true_tau"], tau_s))
    print("T-learner RMSE:", rmse(df["true_tau"], tau_t))

    print("S-learner MAE :", mae(df["true_tau"], tau_s))
    print("T-learner MAE :", mae(df["true_tau"], tau_t))
