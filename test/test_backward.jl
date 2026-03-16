# test/test_backward.jl
using Test
using ChainRulesCore
using ChainRules

include(joinpath(@__DIR__, "..", "src", "tape.jl"))
include(joinpath(@__DIR__, "..", "src", "tracked.jl"))
include(joinpath(@__DIR__, "..", "src", "tape_ops.jl"))
include(joinpath(@__DIR__, "..", "src", "track_call.jl"))
include(joinpath(@__DIR__, "..", "src", "backward.jl"))

function make_tape_with_slots(vals...)
    tape = Tape(TapeEntry[], Any[], Symbol[], Dict{Int,Any}(), false)
    tracked = map(vals) do v
        slot = push_slot!(tape, v)
        Tracked(v, slot, tape)
    end
    return tape, tracked...
end

@testset "accumulate! ignores NoTangent and ZeroTangent" begin
    tape = Tape(TapeEntry[], Any[], Symbol[], Dict{Int,Any}(), false)
    accumulate!(tape, 1, NoTangent())
    accumulate!(tape, 2, ZeroTangent())
    @test !haskey(tape.grad_accum, 1)
    @test !haskey(tape.grad_accum, 2)
end

@testset "accumulate! sums gradients for same slot" begin
    tape = Tape(TapeEntry[], Any[], Symbol[], Dict{Int,Any}(), false)
    accumulate!(tape, 1, [1.0, 2.0])
    accumulate!(tape, 1, [3.0, 4.0])
    @test tape.grad_accum[1] ≈ [4.0, 6.0]
end

@testset "backward! through addition" begin
    # y = x1 + x2; dy/dx1 = 1, dy/dx2 = 1
    tape, tx1, tx2 = make_tape_with_slots([1.0, 2.0], [3.0, 4.0])
    ty = with_tape(tape) do
        track_call(+, tx1, tx2)
    end
    backward!(tape, ty.slot, [1.0, 1.0])
    @test tape.grad_accum[tx1.slot] ≈ [1.0, 1.0]
    @test tape.grad_accum[tx2.slot] ≈ [1.0, 1.0]
end

@testset "backward! through scalar multiply" begin
    # y = x * 3.0; dy/dx = 3
    tape, tx = make_tape_with_slots([2.0, 4.0])
    ty = with_tape(tape) do
        track_call(*, tx, Tracked(3.0, push_slot!(tape, 3.0), tape))
    end
    backward!(tape, ty.slot, [1.0, 1.0])
    @test tape.grad_accum[tx.slot] ≈ [3.0, 3.0]
end
