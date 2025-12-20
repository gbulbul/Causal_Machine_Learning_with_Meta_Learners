import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

def s_learner(df, features):
    X = df[features].copy()
    X["T"] = df["T"]
    y = df["Y"]

    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X, y)

    X1 = X.copy()
    X1["T"] = 1
    X0 = X.copy()
    X0["T"] = 0

    tau_hat = model.predict(X1) - model.predict(X0)
    return tau_hat


def t_learner(df, features):
    treated = df[df["T"] == 1]
    control = df[df["T"] == 0]

    m1 = RandomForestRegressor(n_estimators=200, random_state=42)
    m0 = RandomForestRegressor(n_estimators=200, random_state=42)

    m1.fit(treated[features], treated["Y"])
    m0.fit(control[features], control["Y"])

    tau_hat = m1.predict(df[features]) - m0.predict(df[features])
    return tau_hat
