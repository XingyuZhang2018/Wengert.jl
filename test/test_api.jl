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
