# Physics-Informed Generative Modeling for Synthetic Energy Data

A research mini-project exploring how **physical and domain constraints can be
combined with conditional generative models for electrical energy time series**.

The project progresses from a transparent simulator to physics-aware generation
and finally to domain-informed modeling of real STORM data.

## Research question

> How much physical consistency can be introduced into synthetic energy-data
> generation without unnecessarily sacrificing statistical and temporal fidelity?

## Pipeline

```text
Stage 1                         Stage 2                         Stage 3
Physics simulator  ───────▶    CVAE baseline                  Real STORM data
                               │                              │
                               ├─ Physics-aware CVAE          ├─ CVAE baseline
                               └─ Hard projection             └─ Domain-informed CVAE
```

## Repository structure

```text
storm_physics_genai/
├── README.md
├── requirements.txt
├── stage1_simulator/
│   ├── energy_simulator.py
│   └── README.md
├── stage2_physics_cvae/
│   ├── physics_cvae.py
│   └── README.md
├── stage3_real_storm/
│   ├── physics_cvae_storm.py
│   └── README.md
├── notebooks/
│   ├── stage1_simulator_demo.ipynb
│   └── stage2_physics_cvae_demo.ipynb
├── docs/
└── tests/
```

## Stage 1 — Controlled physics-based simulator

Generates synthetic 15-minute profiles for residential, commercial, and
industrial consumers with weather/calendar effects, PV, EV charging, a radial
network, measurement noise, events, line loading, losses, and approximate
voltage behavior.

The network equations are intentionally simplified and are **not presented as
a full AC power-flow solver**.

## Stage 2 — Physics-aware synthetic-data generation

A conditional VAE jointly generates consumption, PV generation, and net load.
A second model adds differentiable penalties for:

- energy-balance consistency;
- non-negative consumption/PV;
- line-capacity violations;
- lower and upper voltage-limit violations.

A hard-projection baseline provides a comparison between exact constraint
satisfaction and learned soft constraints.

## Stage 3 — Real STORM data

A separate CVAE is trained on real STORM daily time-series windows. Because the
available STORM files do not contain the complete topology and line parameters
needed for power-flow constraints, this stage uses **domain-informed penalties**
rather than claiming full physics-informed learning.

## Installation

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

macOS/Linux:

```bash
source .venv/bin/activate
```

Then:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Reproducibility

All main scripts expose random seeds and write generated outputs to dedicated
result directories. Large datasets, checkpoints, and generated results are
excluded from version control.

## Current research limitations

- The Stage 1/2 radial-grid model is an interpretable approximation, not a
  validated AC power-flow solver.
- Stage 2 uses a dense CVAE rather than an explicitly graph-structured
  generative architecture.
- Stage 3 cannot enforce network power-flow equations without additional
  topology and electrical-parameter data.
- A stronger research extension would compare the CVAE with diffusion/flow
  models and use a validated power-flow layer or graph-based constraint model.

## Project goal

This repository is intended as a compact research prototype for studying
physics-aware synthetic energy data, reproducible model comparison, and future
extensions toward graph-based and privacy-preserving generative modeling.
