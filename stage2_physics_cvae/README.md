# Stage 2 — Physics-aware CVAE on Synthetic Grid Data

Stage 2 trains generative models on the controlled data produced by Stage 1.

## Models

1. **Conditional VAE (CVAE)** — statistical baseline.
2. **Physics-aware CVAE** — CVAE plus differentiable physical penalties.
3. **Hard-projection comparison** — generated variables are post-processed to
   enforce selected identities exactly.

## Generated variables

For every consumer and 15-minute step:

- `consumption_kw`
- `pv_generation_kw`
- `net_load_kw`

The conditioning vector uses calendar information and weather summaries.

## Physics-aware objective

The soft penalty includes:

- consistency of `net_load = consumption - pv_generation`;
- non-negative consumption and PV generation;
- radial-line capacity violations;
- lower and upper voltage-limit violations.

The grid part is differentiable in PyTorch.

## Run

```bash
python stage1_simulator/energy_simulator.py \
  --days 180 --consumers 18 --output stage2_data

python stage2_physics_cvae/physics_cvae.py \
  --data-dir stage2_data \
  --output stage2_cvae_results \
  --epochs 100 \
  --physics-weight 10
```

## Interpretation

This stage is **physics-aware**, not a validated AC power-flow model. Its purpose
is to compare statistical fidelity with progressively stronger physical
consistency under a controlled simulator.
