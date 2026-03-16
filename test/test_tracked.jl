# test/test_tracked.jl
using Test

include(joinpath(@__DIR__, "..", "src", "tape.jl"))
include(joinpath(@__DIR__, "..", "src", "tracked.jl"))

function make_empty_tape()
    Tape(TapeEntry[], Any[], Symbol[], Dict{Int,Any}(), false)
end

@testset "Tracked wraps value" begin
    tape = make_empty_tape()
    x = [1.0, 2.0, 3.0]
    t = Tracked(x, 1, tape)
    @test t.value === x
    @test t.slot == 1
    @test t.tape === tape
end

@testset "Tracked AbstractArray interface" begin
    tape = make_empty_tape()
    x = [1.0 2.0; 3.0 4.0]
    t = Tracked(x, 1, tape)
    @test size(t) == (2, 2)
    @test t[1, 2] == 2.0
    @test length(t) == 4
end

@testset "Tracked wraps non-array type" begin
    tape = make_empty_tape()
    t = Tracked(3.14, 1, tape)
    @test t.value == 3.14
    # scalars don't have size — should not have AbstractArray methods
    @test !hasmethod(size, Tuple{typeof(t)})
end
