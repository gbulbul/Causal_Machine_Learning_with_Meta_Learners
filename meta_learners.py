import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

def s_learner_rf(df, features):
    X = df[features + ["T"]]
    y = df["Y"]

    model = RandomForestRegressor(
        n_estimators=300,
        min_samples_leaf=20,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X, y)

    X1 = X.copy()
    X1["T"] = 1
    X0 = X.copy()
    X0["T"] = 0

    tau_hat = model.predict(X1) - model.predict(X0)
    return tau_hat


def t_learner_rf(df, features):
    df_t = df[df["T"] == 1]
    df_c = df[df["T"] == 0]

    model_t = RandomForestRegressor(
        n_estimators=300,
        min_samples_leaf=20,
        random_state=42,
        n_jobs=-1
    )
    model_c = RandomForestRegressor(
        n_estimators=300,
        min_samples_leaf=20,
        random_state=42,
        n_jobs=-1
    )

    model_t.fit(df_t[features], df_t["Y"])
    model_c.fit(df_c[features], df_c["Y"])

    mu1 = model_t.predict(df[features])
    mu0 = model_c.predict(df[features])

    tau_hat = mu1 - mu0
    return tau_hat


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
