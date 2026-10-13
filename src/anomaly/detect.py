"""Detection and root-cause contribution analysis."""
import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL
from sklearn.ensemble import IsolationForest


def stl_robust_z(series, period=7, threshold=3.5):
    res = STL(series, period=period, robust=True).fit()
    r = res.resid
    mad = np.median(np.abs(r - np.median(r))) * 1.4826
    z = (r - np.median(r)) / mad
    return pd.DataFrame({"value": series, "trend": res.trend, "resid": r, "robust_z": z, "anomaly": z.abs() > threshold})


def isolation_forest(frame, contamination=0.01, seed=0):
    X = frame.fillna(0).values
    clf = IsolationForest(contamination=contamination, random_state=seed).fit(X)
    return pd.Series(clf.predict(X) == -1, index=frame.index)


def contribution(df, kpi, date, dims=("region", "channel", "agent"), baseline_days=28):
    """Compare a date against the same-weekday median of the previous `baseline_days`
    for every segment; return segments ranked by share of the total deviation."""
    d = pd.Timestamp(date)
    hist_dates = [d - pd.Timedelta(days=7 * k) for k in range(1, baseline_days // 7 + 1)]
    out = {}
    for dim in dims:
        cur = df[df.date == d].groupby(dim)[kpi].sum()
        hist = df[df.date.isin(hist_dates)].groupby(dim)[kpi].sum()
        # expected = today's total x each segment's usual share (removes common shocks such as payday)
        base = cur.sum() * hist / hist.sum() if hist.sum() else hist
        delta = (cur - base).fillna(0)
        total = delta.abs().sum() / 2 or 1
        out[dim] = pd.DataFrame({"actual": cur, "expected": base, "delta": delta,
                                 "share_of_delta": delta.abs() / total}).sort_values("delta", key=abs, ascending=False)
    return out


def drill_down(df, kpi, date):
    """Greedy drill-down: region -> channel within region -> agent within region/channel."""
    path, sub = {}, df
    for dim in ["region", "channel", "agent"]:
        c = contribution(sub, kpi, date, dims=(dim,))[dim]
        top = c.index[0]
        path[dim] = (top, float(c.iloc[0].share_of_delta))
        sub = sub[sub[dim] == top]
    return path


def scan_segments(df, kpi, threshold=6.0, return_matrix=False):
    """Run STL + robust z on every region x channel series. National totals hide
    local outages, so production monitoring scans segments and then rolls up."""
    out, mats = [], {}
    total = df.groupby("date")[kpi].sum().asfreq("D")
    for (r, c), g in df.groupby(["region", "channel"]):
        s = g.groupby("date")[kpi].sum().asfreq("D")
        if s.std() == 0:
            continue
        share = s / total  # segment share of national: cancels payday/seasonal shocks common to all
        z = stl_robust_z(share, threshold=threshold)
        mats[f"{r}/{c}"] = z.robust_z
        f = z[z.anomaly]
        out.append(pd.DataFrame({"date": f.index, "region": r, "channel": c, "kpi": kpi, "robust_z": f.robust_z.values}))
    res = pd.concat(out, ignore_index=True) if out else pd.DataFrame()
    return (res, pd.DataFrame(mats)) if return_matrix else res
