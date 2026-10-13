"""Seeded generator for SYNTHETIC daily KPI data (not real operator data).
Grain: date x region x channel x agent. Anomalies are injected with known ground truth."""
import numpy as np
import pandas as pd

REGIONS = ["Nairobi", "Coast", "Rift Valley", "Western", "Nyanza", "Central", "Eastern"]
CHANNELS = ["Agent", "Retail Shop", "USSD Self-serve", "App"]
INJECTED = [  # (date, kpi, region, channel, agent or None, multiplier, description)
    ("2025-03-14", "gross_adds", "Coast", "Agent", None, 0.10, "SIM-registration system outage at Coast agents"),
    ("2025-06-02", "recharges", "Nairobi", "App", None, 0.05, "App payment gateway failure"),
    ("2025-08-21", "agent_float", "Western", "Agent", "AG-W-03", 0.0, "Single super-agent float depletion"),
    ("2025-11-07", "gross_adds", "Rift Valley", "Retail Shop", None, 3.5, "Duplicate activation records (data-quality spike)"),
]


def generate(start="2024-01-01", end="2026-06-30", seed=11, agents_per_region=5):
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, end, freq="D")
    t = np.arange(len(dates))
    weekly = 1 + 0.12 * np.isin(dates.dayofweek, [4, 5])  - 0.15 * (dates.dayofweek == 6)
    payday = 1 + 0.18 * ((dates.day >= 28) | (dates.day <= 2))
    yearly = 1 + 0.08 * np.sin(2 * np.pi * (dates.dayofyear - 330) / 365.25)
    trend = 1 + 0.0004 * t
    ar = np.zeros(len(dates)); e = rng.normal(0, 0.035, len(dates))
    for i in range(1, len(dates)):
        ar[i] = 0.6 * ar[i - 1] + e[i]  # persistent, unmodellable market noise
    base = pd.Series(weekly * payday * yearly * trend * (1 + ar), index=dates)
    rows = []
    rshare = dict(zip(REGIONS, [.30, .11, .18, .11, .10, .11, .09]))
    cshare = dict(zip(CHANNELS, [.50, .25, .15, .10]))
    for r in REGIONS:
        for c in CHANNELS:
            agents = [f"AG-{r[0]}-{i:02d}" for i in range(1, agents_per_region + 1)] if c == "Agent" else ["-"]
            for a in agents:
                share = rshare[r] * cshare[c] / len(agents)
                noise = rng.normal(1, 0.06, (3, len(dates)))
                rows.append(pd.DataFrame({
                    "date": dates, "region": r, "channel": c, "agent": a,
                    "gross_adds": rng.poisson(np.clip(9000 * share * base.values * noise[0], 0, None)),
                    "recharges": (2.5e7 * share * base.values * noise[1]).round(0),
                    "agent_float": (1.2e7 * share * noise[2] * (c == "Agent")).round(0),
                }))
    df = pd.concat(rows, ignore_index=True)
    for d, kpi, r, c, a, mult, _ in INJECTED:
        m = (df.date == d) & (df.region == r) & (df.channel == c)
        if a:
            m &= df.agent == a
        df.loc[m, kpi] = (df.loc[m, kpi] * mult).round(0)
    return df


def national(df, kpi):
    return df.groupby("date")[kpi].sum().asfreq("D")
