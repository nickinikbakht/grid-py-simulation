# Stage 3 — Conditional Generative Modeling on Real STORM Data

Stage 3 trains conditional VAE models directly on real STORM time-series files.

The model uses the STORM channels:

- `S_original`
- `BU_original`

Only complete normal daily windows are used for training. Conditioning includes
weekend/calendar seasonality and station identity.

## Domain-informed constraint

The real STORM files used here do not provide the complete physical network
topology and line parameters required for a full power-flow constraint.
Therefore this stage is deliberately described as **domain-informed**, not
fully physics-informed.

The additional penalty encourages:

- non-negative generated channels;
- realistic statistics of `S_original - BU_original`;
- realistic ramp-rate behavior.

## Expected data layout

```text
storm_data/
└── Train/
    ├── X/
    │   ├── 1.csv
    │   └── ...
    └── y/
        ├── 1.csv
        └── ...
```

## Run

```bash
python stage3_real_storm/physics_cvae_storm.py \
  --train-dir storm_data/Train \
  --output storm_real_cvae_results
```

STORM data are intentionally excluded from Git.
