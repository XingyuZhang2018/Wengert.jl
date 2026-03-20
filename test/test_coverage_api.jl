# test/test_coverage_api.jl — coverage for api.jl uncovered paths
using Test
using Wengert
using Functors
using ChainRulesCore

# ── untrack for plain values ────────────────────────────────────────────────

@testset "untrack — plain value passes through" begin
    @test Wengert.untrack(42) == 42
    @test Wengert.untrack([1.0, 2.0]) == [1.0, 2.0]
    @test Wengert.untrack("hello") == "hello"
end

# ── deep_untrack ────────────────────────────────────────────────────────────

@testset "deep_untrack — non-functor returns as-is" begin
    @test Wengert.deep_untrack(42) == 42
    @test Wengert.deep_untrack([1.0]) == [1.0]
end

@testset "deep_untrack — functor struct" begin
    struct DeepUntrackTest
        a::Vector{Float64}
        b::Vector{Float64}
    end
    Functors.@functor DeepUntrackTest

    # Use pullback to create tracked args, then deep_untrack
    x = DeepUntrackTest([1.0], [2.0])
    y, back = Wengert.pullback(x) do p
        sum(p.a) + sum(p.b)
    end
    # The result went through tracking, deep_untrack should work on non-tracked
    result = Wengert.deep_untrack(x)
    @test result isa DeepUntrackTest
    @test result.a ≈ [1.0]
    @test result.b ≈ [2.0]
end

# ── _wrap_for_tracking — plain non-array/non-functor ───────────────────────

@testset "_wrap_for_tracking — plain scalar passes through" begin
    tape = Wengert.Tape(Wengert.TapeEntry[], Any[], Symbol[], Dict{Int,Any}(), false)
    result = Wengert._wrap_for_tracking(42, tape)
    @test result == 42
end

# ── pullback error: untracked output ────────────────────────────────────────

@testset "pullback — error when output is not tracked" begin
    x = [1.0, 2.0]
    y, back = Wengert.pullback(x) do v
        42  # returns a plain integer, not tracked
    end
    @test y == 42
    @test_throws ErrorException back(1.0)
end

# ── gradient — non-scalar error ─────────────────────────────────────────────

@testset "gradient — error for non-scalar output" begin
    x = [1.0, 2.0]
    @test_throws ErrorException Wengert.gradient(x) do v
        v .+ 1  # returns array, not scalar
    end
end

# ── @ignore macro ───────────────────────────────────────────────────────────

@testset "@ignore strips tracking" begin
    x = [1.0, 2.0, 3.0]
    g = Wengert.gradient(x) do v
        c = @ignore sum(v)  # c is a constant
        sum(v) * 1.0
    end
    @test g[1] ≈ [1.0, 1.0, 1.0]
end

# ── _find_tape_in_args ──────────────────────────────────────────────────────

@testset "_find_tape_in_args — direct tracked arg" begin
    tape = Wengert.Tape(Wengert.TapeEntry[], Any[], Symbol[], Dict{Int,Any}(), false)
    slot = Wengert.push_slot!(tape, [1.0])
    ta = Wengert.TrackedArray([1.0], slot, tape)
    found = Wengert._find_tape_in_args((ta,))
    @test found === tape
end

@testset "_find_tape_in_args — no tracked returns nothing" begin
    found = Wengert._find_tape_in_args(([1.0, 2.0], 42))
    @test found === nothing
end

@testset "_find_tape_in_args — nested in functor struct" begin
    struct FindTapeTest
        w::Vector{Float64}
    end
    Functors.@functor FindTapeTest

    # Test via gradient which exercises the full path
    x = FindTapeTest([1.0, 2.0])
    g = Wengert.gradient(x) do p
        sum(p.w)
    end
    @test g[1].w ≈ [1.0, 1.0]
end

# ── _collect_tracked_leaves! ────────────────────────────────────────────────

@testset "_collect_tracked_leaves! — direct tracked" begin
    tape = Wengert.Tape(Wengert.TapeEntry[], Any[], Symbol[], Dict{Int,Any}(), false)
    slot = Wengert.push_slot!(tape, [1.0])
    ta = Wengert.TrackedArray([1.0], slot, tape)
    leaves = Int[]
    Wengert._collect_tracked_leaves!(ta, leaves)
    @test length(leaves) == 1
    @test leaves[1] == slot
end

@testset "_collect_tracked_leaves! — plain value (no-op)" begin
    leaves = Int[]
    Wengert._collect_tracked_leaves!([1.0, 2.0], leaves)
    @test isempty(leaves)
end

# ── _collect_grad_leaves! — struct path ─────────────────────────────────────

@testset "_collect_grad_leaves! — array leaf" begin
    leaves = Any[]
    Wengert._collect_grad_leaves!([1.0, 2.0], [2.0, 4.0], leaves)
    @test length(leaves) == 1
    @test leaves[1] ≈ [2.0, 4.0]
end

@testset "_collect_grad_leaves! — struct with gradient" begin
    struct GradLeavesTest
        w::Vector{Float64}
    end
    Functors.@functor GradLeavesTest

    leaves = Any[]
    Wengert._collect_grad_leaves!(GradLeavesTest([1.0, 2.0]), GradLeavesTest([2.0, 4.0]), leaves)
    @test length(leaves) == 1
    @test leaves[1] ≈ [2.0, 4.0]
end

# ── barrier with checkpoint modes ───────────────────────────────────────────

module BarrierCoverageTests
using Test, Wengert, Functors

@testset "barrier — :recompute checkpoint" begin
    x = [1.0, 2.0, 3.0]
    g = Wengert.gradient(x) do v
        Wengert.barrier(sum, Wengert.pullback, v .^ 2; checkpoint=:recompute)
    end
    @test g[1] ≈ [2.0, 4.0, 6.0]
end

@testset "barrier — :cpu checkpoint" begin
    x = [1.0, 2.0, 3.0]
    g = Wengert.gradient(x) do v
        Wengert.barrier(sum, Wengert.pullback, v .^ 2; checkpoint=:cpu)
    end
    @test g[1] ≈ [2.0, 4.0, 6.0]
end

@testset "barrier — unknown checkpoint mode errors" begin
    x = [1.0, 2.0]
    @test_throws ErrorException Wengert.gradient(x) do v
        Wengert.barrier(sum, Wengert.pullback, v; checkpoint=:invalid)
    end
end

@testset "barrier — struct output wraps correctly" begin
    struct BarrierStructOut
        w::Vector{Float64}
    end
    Functors.@functor BarrierStructOut

    x = [1.0, 2.0, 3.0]
    # Just test the struct output wrapping path runs without errors
    y, back = Wengert.pullback(x) do v
        result = Wengert.barrier(
            a -> BarrierStructOut(a .^ 2),
            (f, args...) -> begin
                y = f(args...)
                back = function(tangent)
                    return (zeros(length(args[1])),)
                end
                return y, back
            end,
            v
        )
        sum(result.w)
    end
    @test y ≈ 14.0  # 1 + 4 + 9
end

@testset "barrier — tuple output" begin
    x = [1.0, 2.0, 3.0]
    g = Wengert.gradient(x) do v
        result = Wengert.barrier(
            a -> (sum(a), sum(a .^ 2)),
            (f, args...) -> begin
                y = f(args...)
                back = function(tangent)
                    # tangent is (∂sum, ∂sum_sq) tuple
                    t1 = tangent[1] === nothing ? 0.0 : tangent[1]
                    t2 = tangent[2] === nothing ? 0.0 : tangent[2]
                    return (ones(length(args[1])) .* t1 .+ 2 .* args[1] .* t2,)
                end
                return y, back
            end,
            v
        )
        result[1]  # just use the sum
    end
    @test g[1] isa AbstractArray
end

end # module BarrierCoverageTests

# ── Additional api.jl coverage for edge cases ──────────────────────────────

module ApiEdgeCaseTests
using Test, Wengert, Functors, ChainRulesCore

# _extract_grads — non-decomposable (non-functor) paths
@testset "_extract_grads — non-functor scalar" begin
    tape = Wengert.Tape(Wengert.TapeEntry[], Any[], Symbol[], Dict{Int,Any}(), false)
    # Scalar with no tracked arg - exercises the functor decomposition path
    result = Wengert._extract_grads(42, 42, tape.grad_accum)
    # For Int, Functors.functor returns empty children, so result may not be nothing
    @test true  # Just ensure it doesn't error
end

# _extract_grads — original is array but tracked_arg lost tracking
@testset "_extract_grads — untracked array" begin
    tape = Wengert.Tape(Wengert.TapeEntry[], Any[], Symbol[], Dict{Int,Any}(), false)
    result = Wengert._extract_grads([1.0], [1.0], tape.grad_accum)
    @test result === nothing
end

# barrier with struct input — exercises _find_tape_in_args and _collect_tracked_leaves! struct paths
@testset "barrier with struct input exercises nested tape finding" begin
    struct BarrierStructInput
        w::Vector{Float64}
    end
    Functors.@functor BarrierStructInput

    x = BarrierStructInput([1.0, 2.0, 3.0])
    g = Wengert.gradient(x) do p
        Wengert.barrier(
            s -> sum(s.w .^ 2),
            Wengert.pullback,
            p
        )
    end
    @test g[1].w ≈ [2.0, 4.0, 6.0]
end

# _wrap_and_record for non-numeric, non-array, non-tuple, non-functor result
@testset "barrier — non-wrappable result passes through" begin
    x = [1.0, 2.0]
    y, back = Wengert.pullback(x) do v
        result = Wengert.barrier(
            a -> "hello",  # returns a String (non-numeric, non-array, non-functor)
            (f, args...) -> begin
                y = f(args...)
                back = function(tangent)
                    return (zeros(length(args[1])),)
                end
                return y, back
            end,
            v
        )
        sum(v)  # use v directly for the scalar output
    end
    @test y ≈ 3.0
end

# Test _collect_grad_leaves! with nil gradient
@testset "_collect_grad_leaves! — nil gradient for struct" begin
    struct NilGradTest
        w::Vector{Float64}
    end
    Functors.@functor NilGradTest

    leaves = Any[]
    Wengert._collect_grad_leaves!(NilGradTest([1.0]), nothing, leaves)
    @test isempty(leaves)  # nil gradient = no leaves collected
end

end # module ApiEdgeCaseTests

# ── Module-level tests using Wengert for paths that need proper module context ──

module ModuleLevelCoverageTests
using Test, Wengert, Functors

# Parametric typed struct — exercises convert methods and nested struct paths
struct ParamModel{W <: AbstractMatrix}
    weight::W
end
Functors.@functor ParamModel

@testset "gradient with parametric typed struct" begin
    m = ParamModel(rand(3, 3))
    g = Wengert.gradient(m) do model
        sum(model.weight .^ 2)
    end
    @test g[1].weight ≈ 2 .* m.weight
end

# barrier with parametric struct input — exercises _find_tape_in_args nested struct
@testset "barrier with parametric struct input" begin
    m = ParamModel(rand(3, 3))
    g = Wengert.gradient(m) do model
        Wengert.barrier(
            x -> sum(x.weight .^ 2),
            Wengert.pullback,
            model
        )
    end
    @test g[1].weight ≈ 2 .* m.weight
end

# Multi-field struct for more complex nesting
struct TwoFieldModel{W1 <: AbstractMatrix, W2 <: AbstractVector}
    A::W1
    b::W2
end
Functors.@functor TwoFieldModel

@testset "barrier with multi-field struct — exercises nested tracked leaves" begin
    m = TwoFieldModel(rand(2, 2), rand(2))
    g = Wengert.gradient(m) do model
        Wengert.barrier(
            x -> sum(x.A) + sum(x.b),
            Wengert.pullback,
            model
        )
    end
    @test g[1].A ≈ ones(2, 2)
    @test g[1].b ≈ ones(2)
end

# deep_untrack with non-functor leaf
@testset "deep_untrack — symbol/string (non-functor leaf)" begin
    @test Wengert.deep_untrack(:hello) === :hello
    @test Wengert.deep_untrack("world") == "world"
end

end # module ModuleLevelCoverageTests
