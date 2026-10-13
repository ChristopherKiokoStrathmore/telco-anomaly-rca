"""python -m anomaly.run -> reports/figures/*.png, reports/summary.md"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from .data import generate, national, INJECTED
from .detect import scan_segments, stl_robust_z, isolation_forest, drill_down, contribution
from .forecast import evaluate

ROOT = Path(__file__).resolve().parents[2]


def main():
    figs = ROOT / "reports" / "figures"; figs.mkdir(parents=True, exist_ok=True)
    df = generate()
    lines = ["# Results summary (SYNTHETIC DATA)\n", "## Detection vs injected ground truth\n",
             "| Injected date | KPI | National z | Segment scan flagged | Drill-down path (share of deviation) |", "|---|---|---|---|---|"]
    caught = 0
    for d, kpi, *_ , desc in INJECTED:
        nat = national(df, kpi)
        det = stl_robust_z(nat)
        seg = scan_segments(df, kpi)
        hit = seg[seg.date == pd.Timestamp(d)]
        flagged = len(hit) > 0; caught += flagged
        seg_txt = ", ".join(f"{a}/{b} (z={z:.0f})" for a, b, z in zip(hit.region, hit.channel, hit.robust_z)) or "no"
        path = drill_down(df, kpi, d)
        p = " → ".join(f"{v[0]} ({v[1]:.0%})" for v in path.values())
        lines.append(f"| {d} | {kpi} | {det.loc[d,'robust_z']:.1f} (hidden) | {seg_txt} | {p} |")
    # multivariate check
    zmat = pd.concat([scan_segments(df, k, return_matrix=True)[1].add_prefix(k + ":") for k in ["gross_adds", "recharges", "agent_float"]], axis=1)
    n_seg_days = zmat.size
    iso = isolation_forest(zmat.abs(), contamination=0.005)
    fp = sum(len(scan_segments(df, k)) for k in ["gross_adds", "recharges", "agent_float"]) - caught
    lines.append(f"\nSegment scan caught {caught}/{len(INJECTED)} injected incidents; other segment-days flagged: {fp} of {n_seg_days:,} ({fp/n_seg_days:.2%}) for analyst review.")
    lines.append(f"\nIsolationForest on the day x segment |z| matrix flagged {iso.sum()} days; "
                 f"{sum(pd.Timestamp(d) in iso[iso].index for d,*_ in INJECTED)}/{len(INJECTED)} injected dates among them.\n")

    nat = national(df, "gross_adds"); det = stl_robust_z(nat)
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(det.index, det.value, lw=.7, color="#555"); a = det[det.anomaly]
    ax.scatter(a.index, a.value, color="#e4002b", zorder=3, label="Flagged (|robust z| > 3.5)")
    ax.set_title("Daily gross adds, national, with STL + robust-z anomalies (synthetic)"); ax.legend()
    fig.tight_layout(); fig.savefig(figs / "gross_adds_anomalies.png", dpi=150); plt.close(fig)

    c = contribution(df, "gross_adds", INJECTED[0][0])["region"]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(c.index[::-1], c.delta[::-1], color=["#e4002b" if v < 0 else "#9aa5b1" for v in c.delta[::-1]])
    ax.set_title(f"Gross-add deviation by region on {INJECTED[0][0]}"); ax.set_xlabel("Actual - expected (share-adjusted)")
    fig.tight_layout(); fig.savefig(figs / "rca_contribution.png", dpi=150); plt.close(fig)

    train, test, preds, table = evaluate(nat)
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(train.index[-120:], train.iloc[-120:], color="#555", lw=.8, label="Train")
    ax.plot(test.index, test, color="k", lw=1, label="Actual (holdout)")
    for (k, v), col in zip(preds.items(), ["#bbb", "#888", "#e4002b", "#1f77b4"]):
        ax.plot(v.index, v, lw=1, color=col, label=f"{k}: MAPE {table.loc[k,'MAPE %']:.1f}%")
    ax.legend(fontsize=7); ax.set_title("3-month gross-add forecast holdout (synthetic)")
    fig.tight_layout(); fig.savefig(figs / "forecast_holdout.png", dpi=150); plt.close(fig)

    lines += ["## Forecast: 91-day holdout, national gross adds\n", table.to_markdown(), ""]
    md = "\n".join(lines); (ROOT / "reports" / "summary.md").write_text(md); print(md)
    return table, caught


if __name__ == "__main__":
    main()
