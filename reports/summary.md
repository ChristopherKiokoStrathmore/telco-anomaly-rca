# Results summary (SYNTHETIC DATA)

## Detection vs injected ground truth

| Injected date | KPI | National z | Segment scan flagged | Drill-down path (share of deviation) |
|---|---|---|---|---|
| 2025-03-14 | gross_adds | -0.7 (hidden) | Coast/Agent (z=-34), Nairobi/Agent (z=9) | Coast (100%) → Agent (100%) → AG-C-02 (70%) |
| 2025-06-02 | recharges | 1.6 (hidden) | Central/Agent (z=9), Nairobi/App (z=-26) | Nairobi (100%) → App (100%) → - (0%) |
| 2025-08-21 | agent_float | -1.4 (hidden) | Western/Agent (z=-12) | Western (84%) → Agent (0%) → AG-W-03 (100%) |
| 2025-11-07 | gross_adds | 0.4 (hidden) | Nairobi/Agent (z=-8), Rift Valley/Retail Shop (z=51), Rift Valley/USSD Self-serve (z=-8) | Rift Valley (100%) → Retail Shop (100%) → - (0%) |

Segment scan caught 4/4 injected incidents; other segment-days flagged: 299 of 57,456 (0.52%) for analyst review.

IsolationForest on the day x segment |z| matrix flagged 5 days; 1/4 injected dates among them.

## Forecast: 91-day holdout, national gross adds

|                                         |   MAPE % |   WAPE % |
|:----------------------------------------|---------:|---------:|
| Naive (last value)                      |    10.63 |    10.04 |
| Seasonal naive (same weekday last week) |    10.28 |    10.19 |
| ETS (Holt-Winters, weekly)              |     8.96 |     8.65 |
| Gradient boosting (calendar features)   |     4.8  |     4.78 |
