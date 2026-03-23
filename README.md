<p align="center">
  <img src="docs/logo.png" width="220" alt="Spikenaut">
</p>

<h1 align="center">SpikenautLSM.jl</h1>
<p align="center">GPU-accelerated sparse liquid state machine for neuromorphic computing</p>

<p align="center">
  <img src="https://img.shields.io/badge/language-Julia-9558B2" alt="Julia">
  <img src="https://img.shields.io/badge/license-GPL--3.0-orange" alt="GPL-3.0">
</p>

---

Production-grade CUDA-accelerated sparse Liquid State Machine (LSM) with
OU-SDE membrane dynamics, multi-lobe ensemble architecture, cuSPARSE mat-vec,
and STDP covariance learning. Bridges the gap between CPU-only academic SNN
packages and production-scale GPU reservoirs.

## Features

- `SparseBrain` — configurable reservoir: N neurons, connectivity probability, Float16 sparse weights on GPU
- OU-SDE dynamics: `dV = ((V_rest - V)/τ + I_rec + I_ext)dt + σ dW`
- cuSPARSE Float16 sparse mat-vec on GPU (fits 65k neurons in 16 GB VRAM)
- STDP covariance learning with eligibility traces
- 1000-tick rolling spike history buffer (circular, on-GPU)
- `EnsembleBrain` — multi-lobe: multiple reservoirs with different time constants
- Hardware proprioception: global inhibition driven by external stress signals

## Installation

```julia
using Pkg
Pkg.add("SpikenautLSM")
```

## Quick Start

```julia
using SpikenautLSM

# Create a 1024-neuron sparse LSM
brain = SparseBrain(
    n_neurons    = 1024,
    connectivity = 0.1,    # 10% connectivity
    tau          = 20.0,   # ms membrane time constant
    sigma        = 0.05    # noise amplitude
)

# Step the reservoir
for t in 1:T
    external_input = signal[:, t]    # N_inputs × 1
    step!(brain, external_input)
end

readout = brain.spike_history[:, end-99:end]  # last 100 ticks
```

## OU-SDE Membrane Dynamics

```
dV = ((V_rest - V)/τ  +  W_rec·s(t)  +  W_in·x(t)) dt  +  σ dW
```

Discretized as Euler-Maruyama. Spike when `V ≥ θ`; reset to `V_rest`.

*Ornstein & Uhlenbeck (1930); Maass, Natschläger & Markram (2002)*

## STDP Covariance Learning

```
ΔW_ij = η (⟨s_i s_j⟩ - ⟨s_i⟩⟨s_j⟩)
```

Computed on a subsampled 8192-neuron window to avoid O(N²) blow-up.

*Bi & Poo (1998); Hebb (1949)*

## Extracted from Production

Extracted from [Eagle-Lander](https://github.com/rmems/Eagle-Lander), a private
neuromorphic GPU supervisor for crypto mining optimization. The LSM core has been
fully decoupled from market-data ingestion and trading execution so it works
with any time-series application.

## Part of the Spikenaut Ecosystem

| Library | Purpose |
|---------|---------|
| [SpikenautNero.jl](https://github.com/rmems/SpikenautNero.jl) | Multi-lobe relevance scoring |
| [SpikenautDistill.jl](https://github.com/rmems/SpikenautDistill.jl) | Training + FPGA distillation |
| [SpikenautSignals.jl](https://github.com/rmems/SpikenautSignals.jl) | Time-series feature extraction |

## License

GPL-3.0-or-later
