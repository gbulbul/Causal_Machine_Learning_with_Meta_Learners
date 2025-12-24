import numpy as np
import pandas as pd


def simulate_causal_data(
    n=1000,
    p=5,
    seed=42
):
    np.random.seed(seed)

    # Covariates
    X = np.random.normal(0, 1, size=(n, p))
    df = pd.DataFrame(X, columns=[f"X{i+1}" for i in range(p)])

    # Treatment assignment
    propensity = 1 / (1 + np.exp(-0.5 * df["X1"]))
    T = np.random.binomial(1, propensity)
    df["T"] = T

    # -------- TRUE HETEROGENEOUS TREATMENT EFFECT --------
    tau = (
        1.5
        + 2.0 * np.sin(df["X1"])
        - 1.0 * df["X2"]
        + 1.5 * df["X3"] * df["X4"]   # interaction
    )

    df["true_tau"] = tau

    # Baseline outcome (nonlinear)
    mu = (
        2
        + df["X1"] ** 2
        + np.log(np.abs(df["X2"]) + 1)
        - df["X3"]
    )

    # Observed outcome
    noise = np.random.normal(0, 1, size=n)
    Y = mu + T * tau + noise
    df["Y"] = Y

    return df
