# test/test_integration.jl
using Test
using Wengert
import ChainRulesCore: rrule, NoTangent

# --- Fake GPU array to simulate GPU tensors without CUDA ---
struct FakeGPUArray{T} <: AbstractArray{T,1}
    data::Vector{T}
end
Base.size(x::FakeGPUArray) = size(x.data)
Base.getindex(x::FakeGPUArray, i...) = getindex(x.data, i...)
Base.Array(x::FakeGPUArray) = copy(x.data)

# Hook Wengert's GPU detection for FakeGPUArray
Wengert.is_gpu(x::FakeGPUArray) = true
Wengert.to_gpu(x::Vector) = FakeGPUArray(x)

# rrule for sum on FakeGPUArray
function rrule(::typeof(sum), x::FakeGPUArray)
    y = sum(x.data)
    pb(ȳ) = (NoTangent(), FakeGPUArray(fill(convert(eltype(x.data), ȳ), length(x))))
    return y, pb
end

# --- Tests ---

@testset "@checkpoint offloads FakeGPUArray slots to CPU" begin
    x = FakeGPUArray([1.0, 2.0, 3.0])
    tape = Wengert.Tape(Wengert.TapeEntry[], Any[], Symbol[], Dict{Int,Any}(), false)

    Wengert.with_tape(tape) do
        @checkpoint begin
            slot = Wengert.push_slot!(tape, x)
            @test tape.slot_device[slot] == :cpu
            @test tape.slots[slot] isa Vector   # offloaded to plain Array
        end
    end
end

@testset "gradient flows correctly through @checkpoint region" begin
    x = FakeGPUArray([1.0, 2.0, 3.0])
    g = gradient(x) do v
        @checkpoint begin
            sum(v)
        end
    end
    @test g[1] isa FakeGPUArray   # to_gpu restored it
    @test g[1].data ≈ [1.0, 1.0, 1.0]
end
