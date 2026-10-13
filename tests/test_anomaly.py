from anomaly.data import generate, national, INJECTED
from anomaly.detect import stl_robust_z, drill_down, scan_segments
import pandas as pd
from anomaly.forecast import evaluate, mape, wape
import numpy as np

df = generate(start="2025-01-01", end="2026-06-30")


def test_metrics():
    y = np.array([100., 200.]); f = np.array([110., 180.])
    assert abs(mape(y, f) - 10) < 1e-9 and abs(wape(y, f) - 10) < 1e-9


def test_segment_scan_detects_all_injected():
    for d, kpi, r, c, *_ in INJECTED:
        seg = scan_segments(df, kpi)
        assert ((seg.date == pd.Timestamp(d)) & (seg.region == r) & (seg.channel == c)).any()


def test_drill_down_finds_root_cause():
    for d, kpi, r, c, a, *_ in INJECTED:
        path = drill_down(df, kpi, d)
        assert path["region"][0] == r and path["channel"][0] == c
        if a:
            assert path["agent"][0] == a


def test_model_beats_naive():
    _, _, _, t = evaluate(national(df, "gross_adds"))
    best = t.drop(index=[i for i in t.index if "aive" in i])["MAPE %"].min()
    assert best < t.loc["Naive (last value)", "MAPE %"]
    assert best < t.loc["Seasonal naive (same weekday last week)", "MAPE %"]
