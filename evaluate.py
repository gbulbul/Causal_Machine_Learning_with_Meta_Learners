import numpy as np
from simulate_data import simulate_causal_data
from meta_learners import s_learner, t_learner

def rmse(x, y):
    return np.sqrt(np.mean((x - y) ** 2))


if __name__ == "__main__":
    df = simulate_causal_data()
    features = [c for c in df.columns if c.startswith("X")]

    tau_s = s_learner(df, features)
    tau_t = t_learner(df, features)

    print("S-learner RMSE:", rmse(df["true_tau"], tau_s))
    print("T-learner RMSE:", rmse(df["true_tau"], tau_t))
