# Step 1 — STORM Data Understanding and Audit

## My objective

In this first step, I reproduce and inspect the data used in the STORM study on anomaly and change-point detection in power-grid load measurements. I am not training a generative model at this stage. My goal is to understand the data before making modelling decisions.

## Questions I investigate

1. What information is contained in the feature files (`X`) and the label files (`y`)?
2. What is the difference between the measured load (`S_original`) and the bottom-up estimate (`BU_original`)?
3. How common are normal, event/anomaly, and uncertain labels at each station?
4. Do the measurements contain missing values or timestamp inconsistencies?

## Contents

- `STORM_Paper_Reproduction_Step01_Data_English.ipynb` — my reproducible notebook for downloading, loading, auditing, and interactively visualising the training data.

## What the notebook does

1. Downloads the official STORM training split when it is not already available locally.
2. Checks that the station IDs in `X` and `y` agree.
3. Loads one station after checking required columns, duplicate timestamps, and timestamp alignment.
4. Reports the time range, sampling interval, missing values, and label distribution.
5. Selects one training station with labelled events for an interactive Plotly visualisation.

## How I run it

From the repository root:

```bash
jupyter notebook
```

Then open `steps/step_01_data_understanding/STORM_Paper_Reproduction_Step01_Data_English.ipynb` and run the cells from top to bottom.

The first run downloads the official training archive to `storm_data/`. This directory is intentionally excluded from Git because it is downloaded data, not my source code.

## My current interpretation

- `S_original` is the published load measurement at a primary substation.
- `BU_original` is a separately constructed bottom-up estimate of the same aggregate load. It can have a different scale or offset, so I should not use their raw difference as a residual.
- `label = 1` can indicate an anomalous interval or a real operational switching event; it does not automatically mean that the data are wrong.
- `label = 5` means the annotation is uncertain.

## Next step

I will reproduce the preprocessing from the paper: identify invalid observations, robustly align `BU_original` with `S_original`, and build the residual vector $\\delta$. This residual will be the input for later event-aware analysis.

## Limitation

This step uses load time series and labels only. It does not include network topology or line parameters, so it is data understanding rather than a physics-informed experiment.
