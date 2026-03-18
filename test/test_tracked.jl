# test/test_tracked.jl
using Test

include(joinpath(@__DIR__, "..", "src", "tape.jl"))
include(joinpath(@__DIR__, "..", "src", "tracked.jl"))

function make_empty_tape()
    Tape(TapeEntry[], Any[], Symbol[], Dict{Int,Any}(), false)
end

@testset "Tracked wraps scalar value" begin
    tape = make_empty_tape()
    t = Tracked(3.14, 1, tape)
    @test t.value == 3.14
    @test t.slot == 1
    @test t.tape === tape
    # scalars don't have size — should not have AbstractArray methods
    @test !hasmethod(size, Tuple{typeof(t)})
end

@testset "TrackedArray wraps array value" begin
    tape = make_empty_tape()
    x = [1.0, 2.0, 3.0]
    t = TrackedArray(x, 1, tape)
    @test t.value === x
    @test t.slot == 1
    @test t.tape === tape
    @test t isa AbstractArray
    @test t isa AbstractVector{Float64}
end

@testset "TrackedArray AbstractArray interface" begin
    tape = make_empty_tape()
    x = [1.0 2.0; 3.0 4.0]
    t = TrackedArray(x, 1, tape)
    @test size(t) == (2, 2)
    @test t[1, 2] == 2.0
    @test length(t) == 4
    @test eltype(typeof(t)) == Float64
    @test ndims(typeof(t)) == 2
    @test axes(t) == axes(x)
end

@testset "make_tracked dispatches correctly" begin
    tape = make_empty_tape()
    arr = rand(3, 3)
    t_arr = make_tracked(arr, 1, tape)
    @test t_arr isa TrackedArray
    @test t_arr isa AbstractArray{Float64, 2}

    scalar = 42.0
    t_scalar = make_tracked(scalar, 2, tape)
    @test t_scalar isa Tracked
    @test !(t_scalar isa AbstractArray)
end

@testset "AnyTracked union" begin
    tape = make_empty_tape()
    t1 = Tracked(1.0, 1, tape)
    t2 = TrackedArray(rand(2), 2, tape)
    @test t1 isa AnyTracked
    @test t2 isa AnyTracked
end

@testset "TrackedArray satisfies typed field constraint" begin
    struct TestEnv{CT <: AbstractArray{<:Number, 2}}
        C::CT
    end
    tape = make_empty_tape()
    x = rand(3, 3)
    t = TrackedArray(x, 1, tape)
    env = TestEnv(t)
    @test env.C === t
    @test env.C isa AbstractArray{Float64, 2}
end
