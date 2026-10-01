# Stage 1 — STORM-like Physics Simulator

Stage 1 creates synthetic 15-minute energy time series without machine learning.

It simulates residential, commercial, and small industrial consumers; weather
and calendar effects; PV and EV demand; a radial distribution network;
measurement noise; missing values; anomaly/switching-like events; and approximate
line loading, resistive losses, and voltage drop.

## Run

```bash
python stage1_simulator/energy_simulator.py \
  --days 180 \
  --consumers 18 \
  --seed 42 \
  --output stage1_results
```

## STORM-like outputs

The simulator exports `S_original`, `BU_original`, timestamps, missing-value
flags, and labels together with consumer, network, and physical-diagnostic data.

## Physics scope

This stage uses a transparent radial-network approximation rather than a full
AC power-flow solver. It supports both import and reverse-flow signs in the
voltage-drop calculation and checks lower and upper voltage limits.

The approximation is intentional: Stage 1 provides controlled synthetic data
for studying the effect of adding physical constraints to generative models.
