# test/test_track_call.jl
using Test
using ChainRulesCore
using ChainRules

include(joinpath(@__DIR__, "..", "src", "tape.jl"))
include(joinpath(@__DIR__, "..", "src", "tracked.jl"))
include(joinpath(@__DIR__, "..", "src", "tape_ops.jl"))
include(joinpath(@__DIR__, "..", "src", "track_call.jl"))

function make_tape_with_slots(vals...)
    tape = Tape(TapeEntry[], Any[], Symbol[], Dict{Int,Any}(), false)
    tracked = map(vals) do v
        slot = push_slot!(tape, v)
        Tracked(v, slot, tape)
    end
    return tape, tracked...
end

@testset "track_call records entry and returns Tracked" begin
    tape, ta, tb = make_tape_with_slots([1.0, 2.0], [3.0, 4.0])
    result = with_tape(tape) do
        track_call(+, ta, tb)
    end
    @test result isa Tracked
    @test result.value ≈ [4.0, 6.0]
    @test length(tape.entries) == 1
    entry = tape.entries[1]
    @test entry.input_slots == [1, 2]
    @test entry.output_slot == 3
end

@testset "operator + on Tracked arrays" begin
    tape, ta, tb = make_tape_with_slots([1.0], [2.0])
    result = with_tape(tape) do
        ta + tb
    end
    @test result isa Tracked
    @test result.value ≈ [3.0]
end

@testset "operator * (matmul) on Tracked matrices" begin
    tape, ta, tb = make_tape_with_slots([1.0 0.0; 0.0 1.0], [2.0; 3.0])
    result = with_tape(tape) do
        ta * tb
    end
    @test result isa Tracked
    @test result.value ≈ [2.0; 3.0]
end
