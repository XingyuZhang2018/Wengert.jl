# test/test_tape_ops.jl
using Test

include(joinpath(@__DIR__, "..", "src", "tape.jl"))
include(joinpath(@__DIR__, "..", "src", "tracked.jl"))
include(joinpath(@__DIR__, "..", "src", "tape_ops.jl"))

function make_empty_tape()
    Tape(TapeEntry[], Any[], Symbol[], Dict{Int,Any}(), false)
end

@testset "push_slot! on non-GPU value" begin
    tape = make_empty_tape()
    val = [1.0, 2.0]
    id = push_slot!(tape, val)
    @test id == 1
    @test tape.slots[1] === val
    @test tape.slot_device[1] == :gpu   # CPU arrays treated as :gpu (no offload)
end

@testset "push_slot! in checkpoint_mode, plain Array stays :gpu" begin
    tape = make_empty_tape()
    tape.checkpoint_mode = true
    val = [1.0, 2.0]   # plain Array — is_gpu returns false, no offload
    id = push_slot!(tape, val)
    @test tape.slot_device[id] == :gpu  # plain arrays stay :gpu
end

@testset "current_tape returns nothing outside with_tape" begin
    @test current_tape() === nothing
end

@testset "with_tape sets and restores current tape" begin
    tape = make_empty_tape()
    result = with_tape(tape) do
        current_tape()
    end
    @test result === tape
    @test current_tape() === nothing   # restored after block
end

@testset "get_slot_for_backward returns value directly for :gpu slot" begin
    tape = make_empty_tape()
    val = [1.0, 2.0]
    push!(tape.slots, val)
    push!(tape.slot_device, :gpu)
    @test get_slot_for_backward(tape, 1) === val
end

@testset "is_gpu — plain array is false" begin
    @test is_gpu([1.0, 2.0]) == false
    @test is_gpu(42) == false
end

@testset "is_gpu — TrackedArray delegates to value" begin
    tape = make_empty_tape()
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
    tape = make_empty_tape()
    # Manually set up a CPU slot
    push!(tape.slots, [1.0, 2.0])
    push!(tape.slot_device, :cpu)
    # get_slot_for_backward should call to_gpu on it
    result = get_slot_for_backward(tape, 1)
    @test result ≈ [1.0, 2.0]  # to_gpu is identity for plain arrays
end

@testset "get_slot_for_backward — GPU slot returns as-is" begin
    tape = make_empty_tape()
    push!(tape.slots, [3.0, 4.0])
    push!(tape.slot_device, :gpu)
    result = get_slot_for_backward(tape, 1)
    @test result ≈ [3.0, 4.0]
end
