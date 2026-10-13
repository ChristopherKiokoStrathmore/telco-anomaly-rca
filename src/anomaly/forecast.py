"""Gross-add forecasting vs naive baselines on a 3-month holdout."""
import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.ensemble import HistGradientBoostingRegressor


def mape(y, f):
    return float(np.mean(np.abs((y - f) / y)) * 100)


def wape(y, f):
    return float(np.abs(y - f).sum() / np.abs(y).sum() * 100)


def features(idx):
    return pd.DataFrame({"dow": idx.dayofweek, "dom": idx.day, "doy": idx.dayofyear,
                         "payday": ((idx.day >= 28) | (idx.day <= 2)).astype(int),
                         "t": (idx - pd.Timestamp("2024-01-01")).days}, index=idx)


def evaluate(series, horizon=91):
    s = series.copy()
    train, test = s.iloc[:-horizon], s.iloc[-horizon:]
    preds = {
        "Naive (last value)": pd.Series(train.iloc[-1], index=test.index),
        "Seasonal naive (same weekday last week)": pd.Series(np.tile(train.iloc[-7:].values, horizon // 7 + 1)[:horizon], index=test.index),
    }
    ets = ExponentialSmoothing(train, trend="add", damped_trend=True, seasonal="mul", seasonal_periods=7).fit()
    preds["ETS (Holt-Winters, weekly)"] = ets.forecast(horizon)
    gbm = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, random_state=0)
    # GBM on calendar features; learns weekday + payday effects; trend handled via linear detrend
    coef = np.polyfit(features(train.index).t, train.values, 1)
    gbm.fit(features(train.index), train.values - np.polyval(coef, features(train.index).t))
    preds["Gradient boosting (calendar features)"] = pd.Series(
        gbm.predict(features(test.index)) + np.polyval(coef, features(test.index).t), index=test.index)
    table = pd.DataFrame({k: {"MAPE %": mape(test.values, v.values), "WAPE %": wape(test.values, v.values)}
                          for k, v in preds.items()}).T.round(2)
    return train, test, preds, table
