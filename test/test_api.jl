# test/test_api.jl
using Test
using Wengert
using Functors

@testset "pullback — scalar function of a vector" begin
    x = [3.0, 4.0]
    y, back = pullback(x) do v
        sum(v .^ 2)
    end
    @test y ≈ 25.0
    grads = back(1.0)
    @test grads[1] ≈ [6.0, 8.0]   # d/dx sum(x^2) = 2x
end

@testset "pullback — two arguments" begin
    x = [1.0, 2.0]
    y_true = [0.0, 1.0]
    y, back = pullback(x, y_true) do pred, target
        sum((pred .- target) .^ 2)
    end
    grads = back(1.0)
    @test grads[1] ≈ [2.0, 2.0]   # d/dx (x-y)^2 = 2(x-y)
end

@testset "gradient — simple sum of squares" begin
    x = [1.0, 2.0, 3.0]
    g = gradient(x) do v
        sum(v .^ 2)
    end
    @test g[1] ≈ [2.0, 4.0, 6.0]
end

@testset "gradient — struct input via Functors" begin
    struct TwoArrays
        a::Vector{Float64}
        b::Vector{Float64}
    end
    Functors.@functor TwoArrays

    params = TwoArrays([1.0, 2.0], [3.0, 4.0])
    g = gradient(params) do p
        sum(p.a .^ 2) + sum(p.b)
    end
    # ∂/∂a = 2a = [2.0, 4.0], ∂/∂b = ones = [1.0, 1.0]
    @test g[1].a ≈ [2.0, 4.0]
    @test g[1].b ≈ [1.0, 1.0]
end

@testset "gradient — typed struct fields via TrackedArray" begin
    struct TypedEnv{CT <: AbstractArray{<:Number, 2}}
        C::CT
    end
    Functors.@functor TypedEnv

    env = TypedEnv(rand(3, 3))
    g = gradient(env) do e
        sum(e.C .^ 2)
    end
    @test g[1] isa TypedEnv
    @test g[1].C ≈ 2 .* env.C
end

@testset "gradient — multi-field typed struct via TrackedArray" begin
    struct TwoFieldEnv{CT <: AbstractArray{<:Number, 2}, ET <: AbstractArray{<:Number, 3}}
        C::CT
        T::ET
    end
    Functors.@functor TwoFieldEnv

    C = rand(4, 4)
    T = rand(4, 3, 4)
    env = TwoFieldEnv(C, T)
    g = gradient(env) do e
        sum(e.C .^ 2) + sum(e.T)
    end
    @test g[1] isa TwoFieldEnv
    @test g[1].C ≈ 2 .* C
    @test g[1].T ≈ ones(size(T))
end
