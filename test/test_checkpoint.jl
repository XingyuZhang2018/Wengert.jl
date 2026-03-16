# test/test_checkpoint.jl
using Test

include(joinpath(@__DIR__, "..", "src", "tape.jl"))
include(joinpath(@__DIR__, "..", "src", "tracked.jl"))
include(joinpath(@__DIR__, "..", "src", "tape_ops.jl"))
include(joinpath(@__DIR__, "..", "src", "checkpoint.jl"))

function make_empty_tape()
    Tape(TapeEntry[], Any[], Symbol[], Dict{Int,Any}(), false)
end

@testset "@checkpoint sets and restores checkpoint_mode" begin
    tape = make_empty_tape()
    @test tape.checkpoint_mode == false

    with_tape(tape) do
        @checkpoint begin
            @test tape.checkpoint_mode == true
        end
    end

    @test tape.checkpoint_mode == false
end

@testset "@checkpoint restores mode even after exception" begin
    tape = make_empty_tape()
    with_tape(tape) do
        try
            @checkpoint begin
                error("oops")
            end
        catch
        end
    end
    @test tape.checkpoint_mode == false
end

@testset "nested @checkpoint" begin
    tape = make_empty_tape()
    with_tape(tape) do
        @checkpoint begin
            @test tape.checkpoint_mode == true
            @checkpoint begin
                @test tape.checkpoint_mode == true
            end
            @test tape.checkpoint_mode == true
        end
    end
    @test tape.checkpoint_mode == false
end
