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
