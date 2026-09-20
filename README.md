# STORM-like Random Energy Simulator (No ML)

This mini-project creates a synthetic 15-minute electrical-load dataset with the same core table structure as STORM, but without machine learning.

It randomly generates homes, commercial buildings, and small industrial consumers; weather and time effects; PV and EV demand; a radial distribution network; measurement anomalies and switching-like events; missing measurements; approximate line loading, losses, and voltage drop; and publication-ready diagrams.

## Run

```bash
python -m pip install -r requirements.txt
python energy_simulator.py
```

Optional arguments:

```bash
python energy_simulator.py --days 30 --consumers 30 --seed 42 --output results
```

## Important outputs

```text
results/
  Train/X/1.csv                 STORM-style features
  Train/y/1.csv                 STORM-style labels
  consumer_metadata.csv         private consumer characteristics
  consumer_timeseries.csv       demand/PV/net-load data
  network_nodes.csv             substation and consumer nodes
  network_edges.csv             line parameters
  line_timeseries.csv           loading, loss and voltage results
  simulation_config.json        all simulation parameters
  summary.json                  dataset and physical summary
  plots/network.png
  plots/storm_timeseries.png
  plots/residual_and_events.png
  plots/daily_profile.png
  plots/physical_diagnostics.png
```

## STORM-compatible columns

- `M_TIMESTAMP`: UTC timestamp at 15-minute resolution.
- `S_original`: simulated measured apparent power at the substation.
- `BU_original`: noisy bottom-up estimate from consumer meters.
- `missing`: 1 for a deliberately missing/invalid measurement.
- `label`: 0 normal, 1 anomaly/switching event, 5 uncertain boundary.

## Physics scope

This is a transparent educational approximation, not an AC power-flow solver. The radial network uses downstream active power, an assumed power factor, three-phase current, resistive line loss, and approximate voltage drop. A later version can replace `RadialGrid.evaluate()` with pandapower while retaining the generated consumers and time series.

## Later ML stage

The generated 96-point daily profiles can later train a conditional VAE. Physics losses can penalize negative demand, PV at night, voltage violations, and line overloads. Keep `consumer_metadata.csv` private; release only suitable aggregated or synthetic outputs.

