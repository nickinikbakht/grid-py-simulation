# Methodology Notes

## Terminology

- **Physics-based simulator**: Stage 1 uses simplified engineering equations.
- **Physics-aware generative model**: Stage 2 includes differentiable penalties
  derived from the simulator/network approximation.
- **Domain-informed generative model**: Stage 3 uses empirical constraints from
  real STORM channels because full network parameters are not available.

These terms are intentionally separated to avoid overstating physical fidelity.
