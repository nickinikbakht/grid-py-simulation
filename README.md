# Event-Aware Synthetic Energy Data

This repository documents my step-by-step mini project on electricity-load time series, inspired by research on anomaly detection, change-point detection, and synthetic energy data.

My long-term research question is:

> Can event-aware preprocessing help create more realistic synthetic electricity-load time series while preserving meaningful operational behaviour?

I am building this project incrementally. I only describe completed work as completed; planned stages are clearly marked as future work.

## Current status

| Step   | Topic                                        | Status    |
| ------ | -------------------------------------------- | --------- |
| Step 1 | STORM data understanding and audit           | Completed |
| Step 2 | Preprocessing and residual construction      | Planned   |
| Step 3 | Event and change-point analysis              | Planned   |
| Step 4 | Event-aware synthetic time-series baseline   | Planned   |
| Step 5 | Physics-based validation on a public network | Planned   |

## Step 1 — Data understanding and audit

In Step 1, I work with the STORM training dataset to understand its structure before designing any model.

The notebook:

* downloads the official training data;
* checks the correspondence between feature files and label files;
* loads one anonymised station safely;
* audits timestamps, sampling intervals, missing values, and labels;
* visualises measured load, bottom-up load estimates, and labelled events interactively with Plotly.

The main variables are:

* `S_original`: published load measurement at a primary substation;
* `BU_original`: bottom-up estimate of the same aggregate load;
* `missing`: data-quality flag when available;
* `label = 0`: normal operation;
* `label = 1`: anomalous interval or operational switching event;
* `label = 5`: uncertain label.

A key observation is that `S_original` and `BU_original` can differ in scale and offset. Therefore, directly using `S_original - BU_original` would be misleading before preprocessing.

See:

```text
steps/
└── step_01_data_understanding/
    ├── README.md
    └── STORM_Paper_Reproduction_Step01_Data_English.ipynb
```

## How to run Step 1

Create and activate a Python environment:

```bash
python -m venv .venv
source .venv/Scripts/activate
```

Install the required packages:

```bash
pip install pandas plotly jupyter
```

Start Jupyter:

```bash
jupyter notebook
```

Then open and run:

```text
steps/step_01_data_understanding/STORM_Paper_Reproduction_Step01_Data_English.ipynb
```

On the first run, the notebook downloads the STORM training data into `storm_data/`. This downloaded directory is excluded from Git.

## Next step

In Step 2, I will reproduce the preprocessing procedure used in the reference work:

1. identify invalid observations;
2. align the bottom-up estimate with the measured load using robust fitting;
3. construct the residual vector \(\delta\);
4. inspect how events and switching intervals appear in this residual.

This will create a sound basis for later event-aware synthetic time-series modelling.

## Current limitations

This repository does not yet contain a generative model or a physics-informed loss.

The current data provide load measurements and labels, but not full network topology, line parameters, voltage data, or power-flow information. For this reason, Step 1 is a data-understanding and reproducibility exercise, not yet a physics-informed energy-data experiment.

## Why this project matters to me

My previous work on electrical-motor stress monitoring made me interested in the difference between normal transient behaviour, true operational changes, and anomalous signals. This project lets me explore the same general problem in electricity-grid time series, while learning the domain carefully and building a reproducible research workflow.
