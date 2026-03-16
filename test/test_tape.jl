# test/test_tape.jl
using Test

# Load just the types we need for this test
include(joinpath(@__DIR__, "..", "src", "tape.jl"))

@testset "Tape construction" begin
    tape = Tape(TapeEntry[], Any[], Symbol[], Dict{Int,Any}(), false)
    @test isempty(tape.entries)
    @test isempty(tape.slots)
    @test tape.checkpoint_mode == false
end

@testset "TapeEntry fields" begin
    pb = (g) -> (g,)
    entry = TapeEntry(pb, [1, 2], 3)
    @test entry.input_slots == [1, 2]
    @test entry.output_slot == 3
    @test entry.pullback(42) == (42,)
end
