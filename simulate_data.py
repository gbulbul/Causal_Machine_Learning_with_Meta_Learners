import numpy as np
import pandas as pd

def simulate_causal_data(
    n=1000,
    p=10,
    seed=42
):
    """
    Simulate high-dimensional causal data with
    heterogeneous treatment effects.
    """
    np.random.seed(seed)

    X = np.random.normal(0, 1, size=(n, p))
    X = pd.DataFrame(X, columns=[f"X{i}" for i in range(p)])

    # Propensity score
    logits = 0.5 * X["X0"] - 0.3 * X["X1"]
    p_treat = 1 / (1 + np.exp(-logits))
    T = np.random.binomial(1, p_treat)

    # True heterogeneous treatment effect
    tau = 2 + X["X2"] - 0.5 * X["X3"]

    # Baseline outcome
    mu0 = (
        1
        + 0.5 * X["X0"]
        - 0.2 * X["X1"]
        + np.random.normal(0, 1, n)
    )

    Y = mu0 + T * tau

    df = X.copy()
    df["T"] = T
    df["Y"] = Y
    df["true_tau"] = tau

    return df


if __name__ == "__main__":
    df = simulate_causal_data()
    print(df.head())
