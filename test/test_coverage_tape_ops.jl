# test/test_coverage_tape_ops.jl — coverage for tape_ops.jl uncovered paths
using Test

include(joinpath(@__DIR__, "..", "src", "tape.jl"))
include(joinpath(@__DIR__, "..", "src", "tracked.jl"))
include(joinpath(@__DIR__, "..", "src", "tape_ops.jl"))

@testset "is_gpu — plain array is false" begin
    @test is_gpu([1.0, 2.0]) == false
    @test is_gpu(42) == false
end

@testset "is_gpu — TrackedArray delegates to value" begin
    tape = Tape(TapeEntry[], Any[], Symbol[], Dict{Int,Any}(), false)
    x = [1.0, 2.0]
    t = TrackedArray(x, 1, tape)
    @test is_gpu(t) == false
end

@testset "to_gpu — identity for CPU" begin
    x = [1.0, 2.0, 3.0]
    @test to_gpu(x) === x
    @test to_gpu(42) === 42
end

@testset "get_slot_for_backward — CPU slot restores via to_gpu" begin
    tape = Tape(TapeEntry[], Any[], Symbol[], Dict{Int,Any}(), false)
    # Manually set up a CPU slot
    push!(tape.slots, [1.0, 2.0])
    push!(tape.slot_device, :cpu)
    # get_slot_for_backward should call to_gpu on it
    result = get_slot_for_backward(tape, 1)
    @test result ≈ [1.0, 2.0]  # to_gpu is identity for plain arrays
end

@testset "get_slot_for_backward — GPU slot returns as-is" begin
    tape = Tape(TapeEntry[], Any[], Symbol[], Dict{Int,Any}(), false)
    push!(tape.slots, [3.0, 4.0])
    push!(tape.slot_device, :gpu)
    result = get_slot_for_backward(tape, 1)
    @test result ≈ [3.0, 4.0]
end
