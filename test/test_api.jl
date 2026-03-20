# test/test_api.jl — public API tests (pullback, gradient, barrier, @ignore, internals)
using Test
using Wengert
using Functors
using ChainRulesCore
using ChainRulesCore: NoTangent

# ─── pullback ─────────────────────────────────────────────────────────────────

@testset "pullback — scalar function of a vector" begin
    x = [3.0, 4.0]
    y, back = pullback(x) do v
        sum(v .^ 2)
    end
    @test y ≈ 25.0
    grads = back(1.0)
    @test grads[1] ≈ [6.0, 8.0]
end

@testset "pullback — two arguments" begin
    x = [1.0, 2.0]
    y_true = [0.0, 1.0]
    y, back = pullback(x, y_true) do pred, target
        sum((pred .- target) .^ 2)
    end
    grads = back(1.0)
    @test grads[1] ≈ [2.0, 2.0]
end

@testset "pullback — error when output is not tracked" begin
    x = [1.0, 2.0]
    y, back = Wengert.pullback(x) do v
        42  # plain integer, not tracked
    end
    @test y == 42
    @test_throws ErrorException back(1.0)
end

# ─── gradient ─────────────────────────────────────────────────────────────────

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

@testset "gradient — parametric typed struct" begin
    struct ParamModel{W <: AbstractMatrix}
        weight::W
    end
    Functors.@functor ParamModel

    m = ParamModel(rand(3, 3))
    g = Wengert.gradient(m) do model
        sum(model.weight .^ 2)
    end
    @test g[1].weight ≈ 2 .* m.weight
end

@testset "gradient — error for non-scalar output" begin
    x = [1.0, 2.0]
    @test_throws ErrorException Wengert.gradient(x) do v
        v .+ 1
    end
end

@testset "gradient — unused arg returns nothing" begin
    x = [1.0, 2.0]
    y = [3.0, 4.0]
    g = Wengert.gradient(x, y) do a, b
        sum(a .^ 2)
    end
    @test g[1] ≈ [2.0, 4.0]
    @test g[2] === nothing
end

# ─── untrack / deep_untrack ───────────────────────────────────────────────────

@testset "untrack — plain value passes through" begin
    @test Wengert.untrack(42) == 42
    @test Wengert.untrack([1.0, 2.0]) == [1.0, 2.0]
    @test Wengert.untrack("hello") == "hello"
end

@testset "deep_untrack — non-functor returns as-is" begin
    @test Wengert.deep_untrack(42) == 42
    @test Wengert.deep_untrack([1.0]) == [1.0]
    @test Wengert.deep_untrack(:hello) === :hello
    @test Wengert.deep_untrack("world") == "world"
end

@testset "deep_untrack — tuple (children === x path)" begin
    @test Wengert.deep_untrack((1, 2, 3)) === (1, 2, 3)
    @test Wengert.deep_untrack(([1.0, 2.0], [3.0])) == ([1.0, 2.0], [3.0])
end

@testset "deep_untrack — functor struct" begin
    struct DeepUntrackTest
        a::Vector{Float64}
        b::Vector{Float64}
    end
    Functors.@functor DeepUntrackTest

    x = DeepUntrackTest([1.0], [2.0])
    result = Wengert.deep_untrack(x)
    @test result isa DeepUntrackTest
    @test result.a ≈ [1.0]
    @test result.b ≈ [2.0]
end

# ─── @ignore ──────────────────────────────────────────────────────────────────

@testset "@ignore strips tracking" begin
    x = [1.0, 2.0, 3.0]
    g = Wengert.gradient(x) do v
        c = @ignore sum(v)
        sum(v) * 1.0
    end
    @test g[1] ≈ [1.0, 1.0, 1.0]
end

@testset "@ignore uses deep_untrack in gradient" begin
    struct IgnoreTestStruct
        w::Vector{Float64}
    end
    Functors.@functor IgnoreTestStruct

    p = IgnoreTestStruct([1.0, 2.0])
    g = Wengert.gradient(p) do model
        c = @ignore model
        sum(model.w .^ 2)
    end
    @test g[1].w ≈ [2.0, 4.0]
end

# ─── internal helpers ─────────────────────────────────────────────────────────

@testset "_wrap_for_tracking — non-array non-functor passes through" begin
    tape = Wengert.Tape(Wengert.TapeEntry[], Any[], Symbol[], Dict{Int,Any}(), false)
    @test Wengert._wrap_for_tracking(42, tape) == 42
    @test Wengert._wrap_for_tracking(:hello, tape) === :hello
    @test Wengert._wrap_for_tracking("hello", tape) == "hello"
    @test Wengert._wrap_for_tracking((1, 2, 3), tape) === (1, 2, 3)
end

@testset "_find_tape_in_args" begin
    tape = Wengert.Tape(Wengert.TapeEntry[], Any[], Symbol[], Dict{Int,Any}(), false)
    slot = Wengert.push_slot!(tape, [1.0])
    ta = Wengert.TrackedArray([1.0], slot, tape)
    @test Wengert._find_tape_in_args((ta,)) === tape
    @test Wengert._find_tape_in_args(([1.0, 2.0], 42)) === nothing
end

@testset "_find_tape_in_args — nested in functor struct" begin
    struct FindTapeDirectTest{W <: AbstractVector}
        w::W
    end
    Functors.@functor FindTapeDirectTest

    tape = Wengert.Tape(Wengert.TapeEntry[], Any[], Symbol[], Dict{Int,Any}(), false)
    slot = Wengert.push_slot!(tape, [1.0, 2.0])
    ta = Wengert.TrackedArray([1.0, 2.0], slot, tape)
    @test Wengert._find_tape_in_args((FindTapeDirectTest(ta),)) === tape
end

@testset "_find_tape_in_args — struct with no tracked children" begin
    struct FindTapeNoTrack{W}
        w::W
        x::Int
    end
    Functors.@functor FindTapeNoTrack

    model = FindTapeNoTrack([1.0, 2.0], 42)
    @test Wengert._find_tape_in_args((model,)) === nothing
end

@testset "_collect_tracked_leaves!" begin
    tape = Wengert.Tape(Wengert.TapeEntry[], Any[], Symbol[], Dict{Int,Any}(), false)
    slot = Wengert.push_slot!(tape, [1.0])
    ta = Wengert.TrackedArray([1.0], slot, tape)

    leaves = Int[]
    Wengert._collect_tracked_leaves!(ta, leaves)
    @test leaves == [slot]

    leaves2 = Int[]
    Wengert._collect_tracked_leaves!([1.0, 2.0], leaves2)
    @test isempty(leaves2)
end

@testset "_collect_tracked_leaves! — nested struct" begin
    struct CollectLeavesDirectTest{W <: AbstractVector}
        w::W
    end
    Functors.@functor CollectLeavesDirectTest

    tape = Wengert.Tape(Wengert.TapeEntry[], Any[], Symbol[], Dict{Int,Any}(), false)
    slot = Wengert.push_slot!(tape, [1.0])
    ta = Wengert.TrackedArray([1.0], slot, tape)
    leaves = Int[]
    Wengert._collect_tracked_leaves!(CollectLeavesDirectTest(ta), leaves)
    @test leaves == [slot]
end

@testset "_collect_tracked_leaves! — struct with non-tracked children" begin
    struct CollectNoTrackTest{W}
        w::W
    end
    Functors.@functor CollectNoTrackTest

    leaves = Int[]
    Wengert._collect_tracked_leaves!(CollectNoTrackTest([1.0, 2.0]), leaves)
    @test isempty(leaves)
end

@testset "_extract_grads edge cases" begin
    tape = Wengert.Tape(Wengert.TapeEntry[], Any[], Symbol[], Dict{Int,Any}(), false)
    @test Wengert._extract_grads([1.0], [1.0], tape.grad_accum) === nothing
    Wengert._extract_grads(42, 42, tape.grad_accum)  # should not error
    @test Wengert._extract_grads((1, 2), (1, 2), tape.grad_accum) === nothing
end

@testset "_extract_grads — parametric struct" begin
    struct ExtractGradsTest{A <: AbstractVector}
        a::A
    end
    Functors.@functor ExtractGradsTest

    tape = Wengert.Tape(Wengert.TapeEntry[], Any[], Symbol[], Dict{Int,Any}(), false)
    slot = Wengert.push_slot!(tape, [1.0, 2.0])
    ta = Wengert.TrackedArray([1.0, 2.0], slot, tape)
    tape.grad_accum[slot] = [10.0, 20.0]
    result = Wengert._extract_grads(ExtractGradsTest([1.0, 2.0]), ExtractGradsTest(ta), tape.grad_accum)
    @test result isa ExtractGradsTest || result isa NamedTuple
end

@testset "_collect_grad_leaves!" begin
    leaves = Any[]
    Wengert._collect_grad_leaves!([1.0, 2.0], [2.0, 4.0], leaves)
    @test leaves[1] ≈ [2.0, 4.0]
end

@testset "_collect_grad_leaves! — struct with gradient" begin
    struct GradLeavesTest
        w::Vector{Float64}
    end
    Functors.@functor GradLeavesTest
    leaves = Any[]
    Wengert._collect_grad_leaves!(GradLeavesTest([1.0, 2.0]), GradLeavesTest([2.0, 4.0]), leaves)
    @test leaves[1] ≈ [2.0, 4.0]
end

@testset "_collect_grad_leaves! — nil gradient" begin
    struct NilGradTest
        w::Vector{Float64}
    end
    Functors.@functor NilGradTest
    leaves = Any[]
    Wengert._collect_grad_leaves!(NilGradTest([1.0]), nothing, leaves)
    @test isempty(leaves)
end

@testset "_collect_grad_leaves! — non-functor grad for struct" begin
    struct GradLeavesCatchTest{W <: AbstractVector}
        w::W
    end
    Functors.@functor GradLeavesCatchTest
    leaves = Any[]
    Wengert._collect_grad_leaves!(GradLeavesCatchTest([1.0]), (w=[2.0],), leaves)
    @test leaves[1] ≈ [2.0]
end

# ─── TrackedArray module-level tests ──────────────────────────────────────────

@testset "TrackedArray — unsafe_convert method defined" begin
    @test hasmethod(Base.unsafe_convert, Tuple{Type{Ptr{Float64}}, Wengert.TrackedArray{Float64,1,Vector{Float64}}})
end

@testset "TrackedArray — convert via typed struct" begin
    struct ConvertTestEnv{CT <: AbstractArray{Float64, 1}}
        data::CT
    end
    Functors.@functor ConvertTestEnv
    env = ConvertTestEnv([1.0, 2.0, 3.0])
    g = Wengert.gradient(env) do e
        sum(e.data .^ 2)
    end
    @test g[1] isa ConvertTestEnv
    @test g[1].data ≈ [2.0, 4.0, 6.0]
end

@testset "TrackedArray — iterate/show via module" begin
    tape = Wengert.Tape(Wengert.TapeEntry[], Any[], Symbol[], Dict{Int,Any}(), false)
    slot = Wengert.push_slot!(tape, [10.0, 20.0])
    t = Wengert.TrackedArray([10.0, 20.0], slot, tape)
    @test collect(Float64, t) ≈ [10.0, 20.0]
    buf = IOBuffer()
    show(buf, t)
    @test occursin("TrackedArray", String(take!(buf)))
    show(buf, MIME"text/plain"(), t)
    @test occursin("TrackedArray", String(take!(buf)))
end

@testset "is_gpu default via module" begin
    @test Wengert.is_gpu([1.0, 2.0]) == false
    @test Wengert.is_gpu(42) == false
end

@testset "real on TrackedArray" begin
    x = [1.0, 2.0, 3.0]
    y, _ = Wengert.pullback(x) do v
        real(v)
    end
    @test y ≈ [1.0, 2.0, 3.0]
end

# ─── broadcast elementwise via module ─────────────────────────────────────────

_mod_triple(x::Number) = 3.0 * x
ChainRulesCore.rrule(::typeof(_mod_triple), x::Number) =
    3.0 * x, ȳ -> (NoTangent(), 3.0 * ȳ)

@testset "broadcast elementwise via module" begin
    x = [2.0, 3.0, 4.0]
    g = Wengert.gradient(x) do v
        sum(_mod_triple.(v))
    end
    @test g[1] ≈ [3.0, 3.0, 3.0]
end

@testset "broadcast elementwise — multi-arg" begin
    _mod_mul(x::Number, y::Number) = x * y
    ChainRulesCore.rrule(::typeof(_mod_mul), x::Number, y::Number) =
        x * y, ȳ -> (NoTangent(), y * ȳ, x * ȳ)
    x = [1.0, 2.0]; y_arr = [3.0, 4.0]
    g = Wengert.gradient(x, y_arr) do a, b
        sum(_mod_mul.(a, b))
    end
    @test g[1] ≈ [3.0, 4.0]
    @test g[2] ≈ [1.0, 2.0]
end

@testset "broadcast elementwise — same-size arrays" begin
    _mod_scale(x::Number, s::Number) = x * s
    ChainRulesCore.rrule(::typeof(_mod_scale), x::Number, s::Number) =
        x * s, ȳ -> (NoTangent(), s * ȳ, x * ȳ)
    x = [1.0, 2.0, 3.0]; s = [5.0, 5.0, 5.0]
    g = Wengert.gradient(x, s) do arr, scl
        sum(_mod_scale.(arr, scl))
    end
    @test g[1] ≈ [5.0, 5.0, 5.0]
    @test g[2] ≈ [1.0, 2.0, 3.0]
end

@testset "broadcast elementwise — tracked scalar via Ref" begin
    _mod_mul3(x::Number, y::Number) = x * y
    ChainRulesCore.rrule(::typeof(_mod_mul3), x::Number, y::Number) =
        x * y, ȳ -> (NoTangent(), y * ȳ, x * ȳ)
    x = [1.0, 2.0, 3.0]
    g = Wengert.gradient(x) do v
        s = sum(v)
        z = _mod_mul3.(v, Ref(s))
        sum(z)
    end
    @test g[1] isa AbstractArray
end

@testset "broadcast no rrule — returns untracked" begin
    _mod_noop(x::Number) = x + 1000.0
    x = [1.0, 2.0]
    y, _ = Wengert.pullback(x) do v
        _mod_noop.(v)
    end
    @test y ≈ [1001.0, 1002.0]
end

# ─── barrier ──────────────────────────────────────────────────────────────────

module BarrierTests
using Test, Wengert, Functors

@testset "barrier — scalar output" begin
    g = Wengert.gradient([1.0, 2.0, 3.0]) do v
        Wengert.barrier(sum, Wengert.pullback, v .^ 2)
    end
    @test g[1] ≈ [2.0, 4.0, 6.0]
end

@testset "barrier — array output" begin
    g = Wengert.gradient([1.0, 2.0, 3.0]) do v
        y = Wengert.barrier(a -> a .^ 2, Wengert.pullback, v)
        sum(y)
    end
    @test g[1] ≈ [2.0, 4.0, 6.0]
end

@testset "barrier — chained barriers" begin
    A = rand(3, 3)
    g = Wengert.gradient(A) do a
        b = Wengert.barrier(x -> x .* 2, Wengert.pullback, a)
        Wengert.barrier(sum, Wengert.pullback, b .^ 2)
    end
    @test g[1] ≈ 8 .* A
end

@testset "barrier — no tracked args (transparent fallback)" begin
    @test Wengert.barrier(sum, Wengert.pullback, [1.0, 2.0]) == 3.0
end

@testset "barrier — :recompute checkpoint" begin
    g = Wengert.gradient([1.0, 2.0, 3.0]) do v
        Wengert.barrier(sum, Wengert.pullback, v .^ 2; checkpoint=:recompute)
    end
    @test g[1] ≈ [2.0, 4.0, 6.0]
end

@testset "barrier — :cpu checkpoint" begin
    g = Wengert.gradient([1.0, 2.0, 3.0]) do v
        Wengert.barrier(sum, Wengert.pullback, v .^ 2; checkpoint=:cpu)
    end
    @test g[1] ≈ [2.0, 4.0, 6.0]
end

@testset "barrier — unknown checkpoint mode errors" begin
    @test_throws ErrorException Wengert.gradient([1.0, 2.0]) do v
        Wengert.barrier(sum, Wengert.pullback, v; checkpoint=:invalid)
    end
end

@testset "barrier — struct output" begin
    struct BarrierStructOut
        w::Vector{Float64}
    end
    Functors.@functor BarrierStructOut
    x = [1.0, 2.0, 3.0]
    y, _ = Wengert.pullback(x) do v
        result = Wengert.barrier(
            a -> BarrierStructOut(a .^ 2),
            (f, args...) -> (f(args...), _ -> (zeros(length(args[1])),)),
            v)
        sum(result.w)
    end
    @test y ≈ 14.0
end

@testset "barrier — struct output with non-array field" begin
    struct MixedBarrierOut{W <: AbstractVector}
        w::W
        scale::Float64
    end
    Functors.@functor MixedBarrierOut
    x = [1.0, 2.0, 3.0]
    y, _ = Wengert.pullback(x) do v
        result = Wengert.barrier(
            a -> MixedBarrierOut(a .^ 2, 42.0),
            (f, args...) -> (f(args...), _ -> (zeros(length(args[1])),)),
            v)
        sum(result.w)
    end
    @test y ≈ 14.0
end

@testset "barrier — tuple output" begin
    g = Wengert.gradient([1.0, 2.0, 3.0]) do v
        result = Wengert.barrier(
            a -> (sum(a), sum(a .^ 2)),
            (f, args...) -> begin
                y = f(args...)
                back = function(tangent)
                    t1 = tangent[1] === nothing ? 0.0 : tangent[1]
                    t2 = tangent[2] === nothing ? 0.0 : tangent[2]
                    return (ones(length(args[1])) .* t1 .+ 2 .* args[1] .* t2,)
                end
                return y, back
            end, v)
        result[1]
    end
    @test g[1] isa AbstractArray
end

@testset "barrier — non-wrappable result passes through" begin
    y, _ = Wengert.pullback([1.0, 2.0]) do v
        Wengert.barrier(
            a -> "hello",
            (f, args...) -> (f(args...), _ -> (zeros(length(args[1])),)),
            v)
        sum(v)
    end
    @test y ≈ 3.0
end

@testset "barrier — struct input (nested tape finding)" begin
    struct BarrierStructInput
        w::Vector{Float64}
    end
    Functors.@functor BarrierStructInput
    g = Wengert.gradient(BarrierStructInput([1.0, 2.0, 3.0])) do p
        Wengert.barrier(s -> sum(s.w .^ 2), Wengert.pullback, p)
    end
    @test g[1].w ≈ [2.0, 4.0, 6.0]
end

@testset "barrier — parametric struct input" begin
    struct ParamBarrierModel{W <: AbstractMatrix}
        weight::W
    end
    Functors.@functor ParamBarrierModel
    m = ParamBarrierModel(rand(3, 3))
    g = Wengert.gradient(m) do model
        Wengert.barrier(x -> sum(x.weight .^ 2), Wengert.pullback, model)
    end
    @test g[1].weight ≈ 2 .* m.weight
end

@testset "barrier — multi-field struct input" begin
    struct TwoFieldModel{W1 <: AbstractMatrix, W2 <: AbstractVector}
        A::W1
        b::W2
    end
    Functors.@functor TwoFieldModel
    m = TwoFieldModel(rand(2, 2), rand(2))
    g = Wengert.gradient(m) do model
        Wengert.barrier(x -> sum(x.A) + sum(x.b), Wengert.pullback, model)
    end
    @test g[1].A ≈ ones(2, 2)
    @test g[1].b ≈ ones(2)
end

end # module BarrierTests
