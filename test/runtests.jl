using Test
using SpikenautLSM
using CUDA

@testset "SpikenautLSM" begin

    @testset "Package loads" begin
        @test @isdefined(SpikenautLSM)
        @test SpikenautLSM isa Module
    end

    @testset "GPU availability check" begin
        # Package should load regardless of GPU presence
        @test isdefined(SpikenautLSM, :load_sparse_brain)
        @test isdefined(SpikenautLSM, :load_market_lsm)
    end

    if CUDA.functional()
        @testset "GPU: load_market_lsm" begin
            # Wraps the include so CUDA globals allocate on actual GPU
            @test_nowarn SpikenautLSM.load_market_lsm()
            @test @isdefined(N)    # exported from market_lsm
            @test N == 2048
        end
    else
        @info "Skipping GPU tests — no CUDA device available"
        @test_skip "GPU tests skipped (no CUDA)"
    end

end
