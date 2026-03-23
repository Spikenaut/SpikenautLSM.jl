open("/tmp/brain_test.log", "w") do io
    println(io, "Starting Julia Brain test...")
end

# Try loading packages
using ZMQ
using CUDA
using SparseArrays
using Statistics
using LinearAlgebra

open("/tmp/brain_test.log", "a") do io
    println(io, "All packages loaded successfully!")
    println(io, "CUDA device: ", CUDA.name(CUDA.device()))
end

include("sparse_brain.jl")

open("/tmp/brain_test.log", "a") do io
    println(io, "sparse_brain.jl included")
end
