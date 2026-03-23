"""
    SpikenautLSM

GPU-accelerated sparse Liquid State Machine for neuromorphic inference.

Provides two LSM implementations:
- **EnsembleBrain** (`sparse_brain.jl`) — 4-lobe, 65,536-neuron/lobe sparse CUDA LSM
  with OU-SDE dynamics, STDP covariance learning, and rolling 1,000-tick spike history.
  Requires RTX-class GPU with ≥14 GB VRAM.

- **Reference LSM** (`market_lsm.jl`) — 2,048-neuron dense CUDA reservoir with
  16-channel input/output for rapid prototyping.

Both implementations are conditionally loaded when a CUDA GPU is available.
"""
module SpikenautLSM

using CUDA

function __init__()
    if !CUDA.functional()
        @warn "No CUDA-capable GPU found. SpikenautLSM GPU kernels are unavailable. " *
              "Load the source files directly once a GPU is present."
    end
end

"""
    load_sparse_brain()

Include the 65,536-neuron/lobe EnsembleBrain LSM.
Requires CUDA.functional() == true and ≥14 GB VRAM.
"""
function load_sparse_brain()
    if !CUDA.functional()
        error("CUDA GPU required for sparse_brain. CUDA.functional() == false.")
    end
    include(joinpath(@__DIR__, "sparse_brain.jl"))
end

"""
    load_market_lsm()

Include the 2,048-neuron reference LSM reservoir.
Requires CUDA.functional() == true.
"""
function load_market_lsm()
    if !CUDA.functional()
        error("CUDA GPU required for market_lsm. CUDA.functional() == false.")
    end
    include(joinpath(@__DIR__, "market_lsm.jl"))
end

export load_sparse_brain, load_market_lsm

end # module
