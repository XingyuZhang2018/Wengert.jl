# test/test_coverage_track_call.jl — coverage for track_call.jl uncovered paths
using Test
using ChainRulesCore
using ChainRulesCore: unthunk, NoTangent, ZeroTangent
using ChainRules

include(joinpath(@__DIR__, "..", "src", "tape.jl"))
include(joinpath(@__DIR__, "..", "src", "tracked.jl"))
include(joinpath(@__DIR__, "..", "src", "tape_ops.jl"))
include(joinpath(@__DIR__, "..", "src", "track_call.jl"))
include(joinpath(@__DIR__, "..", "src", "backward.jl"))

function make_tape_tracked(vals...)
    tape = Tape(TapeEntry[], Any[], Symbol[], Dict{Int,Any}(), false)
    tracked = map(vals) do v
        slot = push_slot!(tape, v)
        make_tracked(v, slot, tape)
    end
    return tape, tracked...
end

# ── track_call with no rrule returns untracked ──────────────────────────────

_my_noop(x) = x
@testset "track_call — no rrule falls back to untracked" begin
    tape, ta = make_tape_tracked([1.0, 2.0])
    result = with_tape(tape) do
        track_call(_my_noop, ta)
    end
    @test !(result isa AnyTracked)
    @test result ≈ [1.0, 2.0]
end

# ── _ensure_tracked ─────────────────────────────────────────────────────────

@testset "_ensure_tracked — already tracked passes through" begin
    tape, ta = make_tape_tracked([1.0])
    result = _ensure_tracked(ta, tape)
    @test result === ta
end

@testset "_ensure_tracked — plain value gets wrapped" begin
    tape = Tape(TapeEntry[], Any[], Symbol[], Dict{Int,Any}(), false)
    result = _ensure_tracked([1.0, 2.0], tape)
    @test result isa TrackedArray
    @test result.value ≈ [1.0, 2.0]
end

# ── Operator overloads: subtraction ─────────────────────────────────────────

@testset "operator - (TrackedArray, TrackedArray)" begin
    tape, ta, tb = make_tape_tracked([5.0, 6.0], [1.0, 2.0])
    result = with_tape(tape) do; ta - tb; end
    @test result isa AnyTracked
    @test result.value ≈ [4.0, 4.0]
end

@testset "operator - (AnyTracked, plain scalar)" begin
    tape, ta = make_tape_tracked(5.0)
    result = with_tape(tape) do; ta - 1.0; end
    @test result isa AnyTracked
    @test result.value ≈ 4.0
end

@testset "operator - (plain scalar, AnyTracked)" begin
    tape, tb = make_tape_tracked(1.0)
    result = with_tape(tape) do; 5.0 - tb; end
    @test result isa AnyTracked
    @test result.value ≈ 4.0
end

@testset "operator - (TrackedArray, AbstractArray)" begin
    tape, ta = make_tape_tracked([5.0, 6.0])
    result = with_tape(tape) do; ta - [1.0, 2.0]; end
    @test result isa AnyTracked
    @test result.value ≈ [4.0, 4.0]
end

@testset "operator - (AbstractArray, TrackedArray)" begin
    tape, tb = make_tape_tracked([1.0, 2.0])
    result = with_tape(tape) do; [5.0, 6.0] - tb; end
    @test result isa AnyTracked
    @test result.value ≈ [4.0, 4.0]
end

# ── Operator overloads: division ────────────────────────────────────────────

@testset "operator / (tracked scalar, tracked scalar)" begin
    tape, ta, tb = make_tape_tracked(6.0, 3.0)
    result = with_tape(tape) do; ta / tb; end
    @test result isa AnyTracked
    @test result.value ≈ 2.0
end

@testset "operator / (tracked scalar, plain scalar)" begin
    tape, ta = make_tape_tracked(6.0)
    result = with_tape(tape) do; ta / 3.0; end
    @test result isa AnyTracked
    @test result.value ≈ 2.0
end

@testset "operator / (plain scalar, tracked scalar)" begin
    tape, tb = make_tape_tracked(3.0)
    result = with_tape(tape) do; 6.0 / tb; end
    @test result isa AnyTracked
    @test result.value ≈ 2.0
end

# ── Matrix-vector disambiguation ───────────────────────────────────────────

@testset "TrackedMatrix * AbstractVector" begin
    tape, ta = make_tape_tracked([1.0 0.0; 0.0 1.0])
    result = with_tape(tape) do; ta * [2.0, 3.0]; end
    @test result isa AnyTracked
    @test result.value ≈ [2.0, 3.0]
end

@testset "AbstractMatrix * TrackedVector" begin
    tape, tb = make_tape_tracked([2.0, 3.0])
    result = with_tape(tape) do; [1.0 0.0; 0.0 1.0] * tb; end
    @test result isa AnyTracked
    @test result.value ≈ [2.0, 3.0]
end

@testset "TrackedMatrix * TrackedVector" begin
    tape, ta, tb = make_tape_tracked([1.0 0.0; 0.0 1.0], [2.0, 3.0])
    result = with_tape(tape) do; ta * tb; end
    @test result isa AnyTracked
    @test result.value ≈ [2.0, 3.0]
end

@testset "TrackedMatrix * AbstractMatrix" begin
    tape, ta = make_tape_tracked([1.0 2.0; 3.0 4.0])
    result = with_tape(tape) do; ta * [1.0 0.0; 0.0 1.0]; end
    @test result isa AnyTracked
    @test result.value ≈ [1.0 2.0; 3.0 4.0]
end

@testset "AbstractMatrix * TrackedMatrix" begin
    tape, tb = make_tape_tracked([1.0 2.0; 3.0 4.0])
    result = with_tape(tape) do; [1.0 0.0; 0.0 1.0] * tb; end
    @test result isa AnyTracked
    @test result.value ≈ [1.0 2.0; 3.0 4.0]
end

@testset "TrackedMatrix * TrackedMatrix" begin
    tape, ta, tb = make_tape_tracked([1.0 0.0; 0.0 1.0], [1.0 2.0; 3.0 4.0])
    result = with_tape(tape) do; ta * tb; end
    @test result isa AnyTracked
    @test result.value ≈ [1.0 2.0; 3.0 4.0]
end

# ── Unary minus ─────────────────────────────────────────────────────────────

@testset "unary minus on tracked scalar" begin
    tape, ta = make_tape_tracked(5.0)
    result = with_tape(tape) do; -ta; end
    @test result isa AnyTracked
    @test result.value ≈ -5.0
end

# ── real / conj / permutedims ───────────────────────────────────────────────

@testset "real on Tracked{Real} passes through" begin
    tape, ta = make_tape_tracked(3.0)
    result = with_tape(tape) do; real(ta); end
    @test result === ta  # identity for real tracked scalars
end

@testset "conj on TrackedArray" begin
    tape, ta = make_tape_tracked([1.0, 2.0])
    result = with_tape(tape) do; conj(ta); end
    @test result isa AnyTracked
    @test result.value ≈ [1.0, 2.0]
end

@testset "permutedims on TrackedArray" begin
    tape, ta = make_tape_tracked([1.0 2.0; 3.0 4.0])
    result = with_tape(tape) do; permutedims(ta, (2, 1)); end
    @test result isa AnyTracked
    @test result.value ≈ [1.0 3.0; 2.0 4.0]
end

# ── conj rrule for AbstractArray ────────────────────────────────────────────

@testset "rrule(conj, AbstractArray) works" begin
    x = [1.0, 2.0, 3.0]
    y, pb = rrule(conj, x)
    @test y ≈ x
    ȳ = [1.0, 1.0, 1.0]
    grads = pb(ȳ)
    @test grads[1] isa NoTangent
    @test grads[2] ≈ ȳ
end

# ── Broadcast system ────────────────────────────────────────────────────────

@testset "BroadcastStyle for TrackedArray" begin
    @test Base.Broadcast.BroadcastStyle(TrackedArray{Float64,1,Vector{Float64}}) isa TrackedStyle
    @test Base.Broadcast.BroadcastStyle(TrackedStyle(), TrackedStyle()) isa TrackedStyle
    @test Base.Broadcast.BroadcastStyle(TrackedStyle(), Base.Broadcast.DefaultArrayStyle{1}()) isa TrackedStyle
    @test Base.Broadcast.BroadcastStyle(Base.Broadcast.DefaultArrayStyle{1}(), TrackedStyle()) isa TrackedStyle
end

@testset "_extract_broadcast_args" begin
    bc = Base.Broadcast.broadcasted(+, [1.0], [2.0])
    args = _extract_broadcast_args(bc)
    @test args == ([1.0], [2.0])
end

@testset "_unwrap_broadcast_arg dispatches" begin
    tape, ta = make_tape_tracked([1.0, 2.0])
    @test _unwrap_broadcast_arg(ta) ≈ [1.0, 2.0]
    @test _unwrap_broadcast_arg(Ref(3.0)) == 3.0
    @test _unwrap_broadcast_arg(42) == 42
end

@testset "_is_broadcast_scalar dispatches" begin
    tape, ta = make_tape_tracked([1.0])
    @test _is_broadcast_scalar(Ref(1.0)) == true
    @test _is_broadcast_scalar(ta) == false
    @test _is_broadcast_scalar(5.0) == true
    @test _is_broadcast_scalar([1.0]) == false
end

@testset "_find_tape from args" begin
    tape, ta = make_tape_tracked([1.0])
    @test _find_tape((ta,)) === tape
    @test _find_tape(([1.0], [2.0])) === nothing
end

@testset "broadcast with tracked arrays records on tape" begin
    tape, ta = make_tape_tracked([1.0, 2.0, 3.0])
    result = with_tape(tape) do
        ta .^ 2
    end
    @test result isa AnyTracked
    @test result.value ≈ [1.0, 4.0, 9.0]
end

@testset "broadcast with tracked + plain array" begin
    tape, ta = make_tape_tracked([1.0, 2.0])
    result = with_tape(tape) do
        ta .+ [10.0, 20.0]
    end
    @test result isa AnyTracked
    @test result.value ≈ [11.0, 22.0]
end

# ── Nested broadcast ───────────────────────────────────────────────────────

@testset "nested broadcast with TrackedStyle materializes" begin
    tape, ta, tb = make_tape_tracked([1.0, 2.0], [3.0, 4.0])
    result = with_tape(tape) do
        (ta .+ tb) .* [2.0, 2.0]
    end
    @test result isa AnyTracked
    @test result.value ≈ [8.0, 12.0]
end

# ── Element-wise broadcast fallback ─────────────────────────────────────────
# Define a custom function with scalar rrule but NO broadcasted rrule
# This forces the _broadcast_elementwise path in track_call.jl

_custom_double(x::Number) = 2.0 * x
# Scalar rrule — used element-wise
ChainRulesCore.rrule(::WengertRuleConfig, ::typeof(_custom_double), x::Number) =
    2.0 * x, ȳ -> (NoTangent(), 2.0 * ȳ)

include(joinpath(@__DIR__, "..", "src", "api.jl"))

@testset "broadcast elementwise fallback — custom function with scalar rrule" begin
    tape, ta = make_tape_tracked([3.0, 5.0])
    result = with_tape(tape) do
        ta .|> _custom_double   # will use .|> which is broadcast
    end
    # .|> uses broadcast(|>, ta, Ref(_custom_double)) which may not trigger elementwise
    # Use explicit broadcast instead
end

@testset "broadcast elementwise fallback — gradient correctness" begin
    x = [2.0, 3.0, 4.0]
    g = gradient(x) do v
        sum(_custom_double.(v))
    end
    @test g[1] ≈ [2.0, 2.0, 2.0]
end

@testset "broadcast elementwise fallback — two array args" begin
    # Test with two tracked arrays to exercise scalar gradient accumulation
    tape, ta, tb = make_tape_tracked([1.0, 2.0], [3.0, 4.0])
    _custom_add(x::Number, y::Number) = x + y
    ChainRulesCore.rrule(::WengertRuleConfig, ::typeof(_custom_add), x::Number, y::Number) =
        x + y, ȳ -> (NoTangent(), ȳ, ȳ)
    result = with_tape(tape) do
        _custom_add.(ta, tb)
    end
    @test result isa AnyTracked
    @test result.value ≈ [4.0, 6.0]
end

@testset "broadcast elementwise — array input" begin
    tape, ta = make_tape_tracked([1.0, 2.0, 3.0])
    result = with_tape(tape) do
        _custom_double.(ta)
    end
    @test result isa AnyTracked
    @test result.value ≈ [2.0, 4.0, 6.0]
end

@testset "broadcast elementwise — full gradient through backward" begin
    x = [1.0, 2.0]
    g = gradient(x) do v
        y = _custom_double.(v)
        sum(y)
    end
    @test g[1] ≈ [2.0, 2.0]
end

# Define a function with NO rrule at all to test untracked broadcast fallback
_no_rrule_fn(x::Number) = x + 100.0

@testset "broadcast elementwise — no rrule returns untracked" begin
    tape, ta = make_tape_tracked([1.0, 2.0])
    result = with_tape(tape) do
        _no_rrule_fn.(ta)
    end
    # Should return plain array since no rrule exists
    @test result ≈ [101.0, 102.0]
end

# ── real on AnyTracked (non-real or general) ───────────────────────────────

@testset "sum on TrackedArray" begin
    tape, ta = make_tape_tracked([1.0, 2.0, 3.0])
    result = with_tape(tape) do
        sum(ta)
    end
    @test result isa AnyTracked
    @test result.value ≈ 6.0
end

# ── Division operator disambiguation for matrices ──────────────────────────

@testset "TrackedMatrix - AbstractMatrix" begin
    tape, ta = make_tape_tracked([1.0 2.0; 3.0 4.0])
    result = with_tape(tape) do; ta - [0.0 1.0; 1.0 0.0]; end
    @test result isa AnyTracked
    @test result.value ≈ [1.0 1.0; 2.0 4.0]
end

@testset "AbstractMatrix - TrackedMatrix" begin
    tape, tb = make_tape_tracked([0.0 1.0; 1.0 0.0])
    result = with_tape(tape) do; [1.0 2.0; 3.0 4.0] - tb; end
    @test result isa AnyTracked
    @test result.value ≈ [1.0 1.0; 2.0 4.0]
end

# ── _is_broadcast_scalar coverage via broadcast ────────────────────────────

@testset "broadcast with Ref scalar" begin
    tape, ta = make_tape_tracked([1.0, 2.0, 3.0])
    result = with_tape(tape) do
        ta .+ Ref(10.0)
    end
    @test result isa AnyTracked
    @test result.value ≈ [11.0, 12.0, 13.0]
end
