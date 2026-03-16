# YAADE.jl Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a tape-based reverse-mode AD engine in Julia that supports explicit CPU gradient checkpointing and is compatible with ChainRulesCore rules.

**Architecture:** A flat Wengert-list tape records operations during the forward pass. Each `Tracked{T}` value carries a slot ID into the tape's value store. The `@checkpoint` macro sets a flag that causes `push_slot!` to offload GPU tensors to CPU immediately; `backward!` retrieves them on demand.

**Tech Stack:** Julia 1.9+, ChainRulesCore.jl, Functors.jl, CUDA.jl (optional, for GPU tests), stdlib Test

---

## File Map

```
YAADE.jl/
├── Project.toml
├── src/
│   ├── YAADE.jl          # module entry, re-exports
│   ├── tape.jl           # Tape, TapeEntry
│   ├── tracked.jl        # Tracked{T} + AbstractArray interface
│   ├── tape_ops.jl       # push_slot!, current_tape, with_tape, get_slot_for_backward
│   ├── track_call.jl     # track_call, Base operator overloads
│   ├── checkpoint.jl     # @checkpoint macro
│   ├── backward.jl       # backward!, accumulate!
│   └── api.jl            # pullback, gradient, untrack
└── test/
    ├── runtests.jl
    ├── test_tape.jl
    ├── test_tracked.jl
    ├── test_tape_ops.jl
    ├── test_track_call.jl
    ├── test_checkpoint.jl
    ├── test_backward.jl
    └── test_api.jl
```

---

### Task 1: Package Scaffolding

**Files:**
- Create: `Project.toml`
- Create: `src/YAADE.jl`
- Create: `test/runtests.jl`

**Step 1: Create Project.toml**

```toml
name = "YAADE"
uuid = "00000000-0000-0000-0000-000000000001"
authors = []
version = "0.1.0"

[deps]
ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
Functors = "d9f16b24-f501-4c13-a1f2-28368ffc5196"

[compat]
ChainRulesCore = "1"
Functors = "0.4"
julia = "1.9"

[extras]
Test = "8dfed614-e22c-358a-9a83-a03f1d5d13c3"

[targets]
test = ["Test"]
```

**Step 2: Create src/YAADE.jl (empty module for now)**

```julia
module YAADE

using ChainRulesCore
using Functors

include("tape.jl")
include("tracked.jl")
include("tape_ops.jl")
include("track_call.jl")
include("checkpoint.jl")
include("backward.jl")
include("api.jl")

export Tracked, Tape
export pullback, gradient
export @checkpoint

end
```

**Step 3: Create test/runtests.jl**

```julia
using Test

@testset "YAADE" begin
    include("test_tape.jl")
    include("test_tracked.jl")
    include("test_tape_ops.jl")
    include("test_track_call.jl")
    include("test_checkpoint.jl")
    include("test_backward.jl")
    include("test_api.jl")
end
```

**Step 4: Instantiate the project**

```bash
cd "D:/1 - research/1.9 - AutoDiff/YAADE.jl"
julia --project=. -e "using Pkg; Pkg.instantiate()"
```
Expected: Downloads ChainRulesCore and Functors, no errors.

**Step 5: Commit**

```bash
git add Project.toml src/YAADE.jl test/runtests.jl
git commit -m "chore: scaffold YAADE.jl package"
```

---

### Task 2: Tape and TapeEntry Types

**Files:**
- Create: `src/tape.jl`
- Create: `test/test_tape.jl`

**Step 1: Write the failing test**

```julia
# test/test_tape.jl
using Test
using YAADE: TapeEntry, Tape

@testset "Tape construction" begin
    tape = Tape(TapeEntry[], Any[], Symbol[], Dict{Int,Any}(), false)
    @test isempty(tape.entries)
    @test isempty(tape.slots)
    @test tape.checkpoint_mode == false
end

@testset "TapeEntry fields" begin
    pb = (g) -> (g,)
    entry = TapeEntry(pb, [1, 2], 3)
    @test entry.input_slots == [1, 2]
    @test entry.output_slot == 3
    @test entry.pullback(42) == (42,)
end
```

**Step 2: Run test to verify it fails**

```bash
julia --project=. test/runtests.jl
```
Expected: ERROR — `TapeEntry` not defined.

**Step 3: Implement src/tape.jl**

```julia
mutable struct TapeEntry
    pullback::Any
    input_slots::Vector{Int}
    output_slot::Int
end

mutable struct Tape
    entries::Vector{TapeEntry}
    slots::Vector{Any}
    slot_device::Vector{Symbol}  # :gpu | :cpu
    grad_accum::Dict{Int,Any}
    checkpoint_mode::Bool
end
```

**Step 4: Run test to verify it passes**

```bash
julia --project=. test/runtests.jl
```
Expected: `Tape construction` and `TapeEntry fields` PASS.

**Step 5: Commit**

```bash
git add src/tape.jl test/test_tape.jl
git commit -m "feat: add Tape and TapeEntry types"
```

---

### Task 3: Tracked{T} Type

**Files:**
- Create: `src/tracked.jl`
- Create: `test/test_tracked.jl`

**Step 1: Write the failing test**

```julia
# test/test_tracked.jl
using Test
using YAADE: Tape, TapeEntry, Tracked

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
```

**Step 2: Run test to verify it fails**

```bash
julia --project=. test/runtests.jl
```
Expected: ERROR — `Tracked` not defined.

**Step 3: Implement src/tracked.jl**

```julia
struct Tracked{T}
    value::T
    slot::Int
    tape::Tape
end

# AbstractArray interface — only for array-backed Tracked values
Base.size(x::Tracked{<:AbstractArray}) = size(x.value)
Base.getindex(x::Tracked{<:AbstractArray}, i...) = getindex(x.value, i...)
Base.length(x::Tracked{<:AbstractArray}) = length(x.value)
Base.IndexStyle(::Type{<:Tracked{T}}) where {T<:AbstractArray} = IndexStyle(T)
Base.eltype(::Type{Tracked{T}}) where {T<:AbstractArray} = eltype(T)
Base.ndims(::Type{Tracked{T}}) where {T<:AbstractArray} = ndims(T)
```

**Step 4: Run test to verify it passes**

```bash
julia --project=. test/runtests.jl
```
Expected: All `Tracked` tests PASS.

**Step 5: Commit**

```bash
git add src/tracked.jl test/test_tracked.jl
git commit -m "feat: add Tracked{T} type with AbstractArray interface"
```

---

### Task 4: Tape Operations

**Files:**
- Create: `src/tape_ops.jl`
- Create: `test/test_tape_ops.jl`

**Step 1: Write the failing test**

```julia
# test/test_tape_ops.jl
using Test
using YAADE: Tape, TapeEntry, Tracked
using YAADE: push_slot!, current_tape, with_tape, get_slot_for_backward

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

@testset "push_slot! in checkpoint_mode moves CPU array (no CUDA)" begin
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
```

**Step 2: Run test to verify it fails**

```bash
julia --project=. test/runtests.jl
```
Expected: ERROR — `push_slot!` not defined.

**Step 3: Implement src/tape_ops.jl**

```julia
# Task-local tape — thread-safe, no global state
const TAPE_KEY = :yaade_active_tape

current_tape() = get(task_local_storage(), TAPE_KEY, nothing)

function with_tape(f::Function, tape::Tape)
    task_local_storage(TAPE_KEY, tape) do
        f()
    end
end

# GPU detection — works for plain Arrays and CuArrays
is_gpu(x) = false   # overridden by CUDA extension

function push_slot!(tape::Tape, value)
    if tape.checkpoint_mode && is_gpu(value)
        cpu_val = Array(value)   # offload to CPU
        push!(tape.slots, cpu_val)
        push!(tape.slot_device, :cpu)
    else
        push!(tape.slots, value)
        push!(tape.slot_device, :gpu)
    end
    return length(tape.slots)
end

function get_slot_for_backward(tape::Tape, id::Int)
    if tape.slot_device[id] === :cpu
        return to_gpu(tape.slots[id])   # restore to GPU
    end
    return tape.slots[id]
end

# to_gpu stub — overridden by CUDA extension
to_gpu(x) = x
```

**Step 4: Run test to verify it passes**

```bash
julia --project=. test/runtests.jl
```
Expected: All tape_ops tests PASS.

**Step 5: Commit**

```bash
git add src/tape_ops.jl test/test_tape_ops.jl
git commit -m "feat: add tape slot management and task-local tape storage"
```

---

### Task 5: track_call and Operator Overloading

**Files:**
- Create: `src/track_call.jl`
- Create: `test/test_track_call.jl`

**Step 1: Write the failing test**

```julia
# test/test_track_call.jl
using Test
using ChainRulesCore
using YAADE: Tape, TapeEntry, Tracked, push_slot!, with_tape
using YAADE: track_call

function make_tape_with_slots(vals...)
    tape = Tape(TapeEntry[], Any[], Symbol[], Dict{Int,Any}(), false)
    tracked = map(vals) do v
        slot = push_slot!(tape, v)
        Tracked(v, slot, tape)
    end
    return tape, tracked...
end

@testset "track_call records entry and returns Tracked" begin
    tape, ta, tb = make_tape_with_slots([1.0, 2.0], [3.0, 4.0])
    result = with_tape(tape) do
        track_call(+, ta, tb)
    end
    @test result isa Tracked
    @test result.value ≈ [4.0, 6.0]
    @test length(tape.entries) == 1
    entry = tape.entries[1]
    @test entry.input_slots == [1, 2]
    @test entry.output_slot == 3
end

@testset "operator + on Tracked arrays" begin
    tape, ta, tb = make_tape_with_slots([1.0], [2.0])
    result = with_tape(tape) do
        ta + tb
    end
    @test result isa Tracked
    @test result.value ≈ [3.0]
end

@testset "operator * (matmul) on Tracked matrices" begin
    tape, ta, tb = make_tape_with_slots([1.0 0.0; 0.0 1.0], [2.0; 3.0])
    result = with_tape(tape) do
        ta * tb
    end
    @test result isa Tracked
    @test result.value ≈ [2.0; 3.0]
end
```

**Step 2: Run test to verify it fails**

```bash
julia --project=. test/runtests.jl
```
Expected: ERROR — `track_call` not defined.

**Step 3: Implement src/track_call.jl**

```julia
struct YAADERuleConfig <: RuleConfig{Union{}} end

function track_call(f, args::Tracked...)
    tape = args[1].tape
    raw_args = map(a -> a.value, args)

    # Attempt rrule; if none exists, fall through without tracking
    result = rrule_via_ad(YAADERuleConfig(), f, raw_args...)
    if result === nothing
        # No rrule — call f directly, return untracked
        return f(raw_args...)
    end
    y, pb = result

    out_slot = push_slot!(tape, y)
    push!(tape.entries, TapeEntry(pb, Int[a.slot for a in args], out_slot))
    return Tracked(y, out_slot, tape)
end

# Promote mixed Tracked/non-Tracked calls: wrap non-Tracked in untracked slot
function _ensure_tracked(x::Tracked, tape) = x
function _ensure_tracked(x, tape)
    slot = push_slot!(tape, x)
    Tracked(x, slot, tape)
end

# Overload common Base operations
for op in (:+, :-, :*, :/)
    @eval begin
        Base.$op(a::Tracked, b::Tracked) = track_call($op, a, b)
        Base.$op(a::Tracked, b) = track_call($op, a, _ensure_tracked(b, a.tape))
        Base.$op(a, b::Tracked) = track_call($op, _ensure_tracked(a, b.tape), b)
    end
end

Base.:-(a::Tracked) = track_call(-, a)

# Linear algebra
import LinearAlgebra
LinearAlgebra.mul!(C, a::Tracked, b::Tracked) = track_call(*, a, b)
```

**Step 4: Run test to verify it passes**

```bash
julia --project=. test/runtests.jl
```
Expected: All `track_call` tests PASS.

**Step 5: Commit**

```bash
git add src/track_call.jl test/test_track_call.jl
git commit -m "feat: add track_call and operator overloads for Tracked"
```

---

### Task 6: `@checkpoint` Macro

**Files:**
- Create: `src/checkpoint.jl`
- Create: `test/test_checkpoint.jl`

**Step 1: Write the failing test**

```julia
# test/test_checkpoint.jl
using Test
using YAADE: Tape, TapeEntry, Tracked, push_slot!, with_tape
using YAADE: @checkpoint

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
```

**Step 2: Run test to verify it fails**

```bash
julia --project=. test/runtests.jl
```
Expected: ERROR — `@checkpoint` not defined.

**Step 3: Implement src/checkpoint.jl**

```julia
macro checkpoint(expr)
    quote
        local _tape = current_tape()
        if _tape === nothing
            error("@checkpoint used outside of pullback/gradient context")
        end
        local _prev = _tape.checkpoint_mode
        _tape.checkpoint_mode = true
        try
            $(esc(expr))
        finally
            _tape.checkpoint_mode = _prev
        end
    end
end
```

**Step 4: Run test to verify it passes**

```bash
julia --project=. test/runtests.jl
```
Expected: All `@checkpoint` tests PASS.

**Step 5: Commit**

```bash
git add src/checkpoint.jl test/test_checkpoint.jl
git commit -m "feat: add @checkpoint macro with save/restore and exception safety"
```

---

### Task 7: Backward Pass

**Files:**
- Create: `src/backward.jl`
- Create: `test/test_backward.jl`

**Step 1: Write the failing test**

```julia
# test/test_backward.jl
using Test
using ChainRulesCore
using YAADE: Tape, TapeEntry, Tracked, push_slot!, with_tape, track_call
using YAADE: backward!, accumulate!

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
    # y = 3 * x; dy/dx = 3
    tape, tx = make_tape_with_slots([2.0, 4.0])
    ty = with_tape(tape) do
        track_call(*, tx, Tracked(3.0, push_slot!(tape, 3.0), tape))
    end
    backward!(tape, ty.slot, [1.0, 1.0])
    @test tape.grad_accum[tx.slot] ≈ [3.0, 3.0]
end
```

**Step 2: Run test to verify it fails**

```bash
julia --project=. test/runtests.jl
```
Expected: ERROR — `backward!` not defined.

**Step 3: Implement src/backward.jl**

```julia
function accumulate!(tape::Tape, slot::Int, grad)
    grad isa NoTangent  && return
    grad isa ZeroTangent && return
    existing = get(tape.grad_accum, slot, nothing)
    tape.grad_accum[slot] = isnothing(existing) ? grad : add!!(existing, grad)
end

function backward!(tape::Tape, output_slot::Int, ȳ)
    accumulate!(tape, output_slot, ȳ)

    for entry in Iterators.reverse(tape.entries)
        g_out = get(tape.grad_accum, entry.output_slot, nothing)
        g_out === nothing && continue   # no gradient flows through this entry

        # inputs may need to be retrieved from CPU (checkpoint)
        input_vals = [get_slot_for_backward(tape, s) for s in entry.input_slots]

        # call the ChainRulesCore pullback; first return is grad for f itself
        raw_grads = entry.pullback(g_out)

        # raw_grads[1] is ∂f (usually NoTangent), rest match input_slots
        # Note: rrule returns (∂self, ∂arg1, ∂arg2, ...) — skip index 1 (∂self)
        for (slot, g) in zip(entry.input_slots, Iterators.drop(raw_grads, 1))
            accumulate!(tape, slot, g)
        end
    end
end
```

> **Note on pullback indexing:** ChainRulesCore `rrule(f, args...)` returns `(y, pb)` where `pb(ȳ)` returns `(∂f, ∂arg1, ∂arg2, ...)`. The first element is the gradient w.r.t. `f` itself (almost always `NoTangent()`). We skip it by using `Iterators.drop(raw_grads, 1)` and zip with `input_slots` which only contains the argument slots.

**Step 4: Run test to verify it passes**

```bash
julia --project=. test/runtests.jl
```
Expected: All backward tests PASS.

**Step 5: Commit**

```bash
git add src/backward.jl test/test_backward.jl
git commit -m "feat: add backward! and accumulate! for tape replay"
```

---

### Task 8: `pullback` and `gradient` API

**Files:**
- Create: `src/api.jl`
- Create: `test/test_api.jl`

**Step 1: Write the failing test**

```julia
# test/test_api.jl
using Test
using ChainRulesCore
using Functors
using YAADE

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
    # A simple struct with two array fields
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
```

**Step 2: Run test to verify it fails**

```bash
julia --project=. test/runtests.jl
```
Expected: ERROR — `pullback` not defined.

**Step 3: Implement src/api.jl**

```julia
# Strip Tracked wrapper from a value, recursively for Functors-compatible types
untrack(x::Tracked) = x.value
untrack(x) = x

function pullback(f, args...)
    tape = Tape(TapeEntry[], Any[], Symbol[], Dict{Int,Any}(), false)

    # Wrap all leaf arrays in Tracked, recording their slots
    tracked_args = map(args) do arg
        Functors.fmap(arg; exclude = x -> x isa Number && !(x isa AbstractArray)) do leaf
            if leaf isa AbstractArray
                slot = push_slot!(tape, leaf)
                Tracked(leaf, slot, tape)
            else
                leaf
            end
        end
    end

    # Run f, capturing tape entries
    result = with_tape(tape) do
        f(tracked_args...)
    end

    y = untrack(result)

    function back(ȳ)
        if !(result isa Tracked)
            error("pullback: function output was not tracked — ensure it calls operations with rrule")
        end
        backward!(tape, result.slot, ȳ)

        # Reconstruct gradient structs matching the input shapes
        return map(tracked_args) do arg
            Functors.fmap(arg) do leaf
                if leaf isa Tracked
                    get(tape.grad_accum, leaf.slot, nothing)
                else
                    nothing
                end
            end
        end
    end

    return y, back
end

function gradient(f, args...)
    y, back = pullback(f, args...)
    y isa Number || error("gradient: function must return a scalar, got $(typeof(y))")
    return back(one(y))
end
```

**Step 4: Run test to verify it passes**

```bash
julia --project=. test/runtests.jl
```
Expected: All API tests PASS.

**Step 5: Commit**

```bash
git add src/api.jl test/test_api.jl
git commit -m "feat: add pullback and gradient public API with Functors struct support"
```

---

### Task 9: Integration Test — Checkpoint + CPU Offload (CPU-only simulation)

**Files:**
- Create: `test/test_integration.jl`

Add `include("test_integration.jl")` to `test/runtests.jl`.

**Step 1: Write the integration test**

This test simulates checkpoint behaviour on CPU arrays (no CUDA required). It verifies that `@checkpoint` marks slots as `:cpu` when `is_gpu` returns `true` via a monkey-patch.

```julia
# test/test_integration.jl
using Test
using YAADE
import YAADE: is_gpu, to_gpu, push_slot!, Tape, TapeEntry

# Simulate GPU arrays with a wrapper type
struct FakeGPUArray{T} <: AbstractArray{T,1}
    data::Vector{T}
end
Base.size(x::FakeGPUArray) = size(x.data)
Base.getindex(x::FakeGPUArray, i...) = getindex(x.data, i...)
Base.Array(x::FakeGPUArray) = copy(x.data)

# Override is_gpu and to_gpu for FakeGPUArray
YAADE.is_gpu(x::FakeGPUArray) = true
YAADE.to_gpu(x::Vector) = FakeGPUArray(x)

# ChainRulesCore rrule for sum on FakeGPUArray
import ChainRulesCore: rrule, NoTangent
function rrule(::typeof(sum), x::FakeGPUArray)
    y = sum(x.data)
    pb(ȳ) = (NoTangent(), FakeGPUArray(fill(ȳ, length(x))))
    return y, pb
end

@testset "@checkpoint offloads FakeGPUArray slots to CPU" begin
    x = FakeGPUArray([1.0, 2.0, 3.0])
    tape = Tape(TapeEntry[], Any[], Symbol[], Dict{Int,Any}(), false)

    YAADE.with_tape(tape) do
        @checkpoint begin
            slot = push_slot!(tape, x)
            @test tape.slot_device[slot] == :cpu
            @test tape.slots[slot] isa Vector   # offloaded to plain Array
        end
    end
end

@testset "gradient flows correctly through @checkpoint region" begin
    x = FakeGPUArray([1.0, 2.0, 3.0])
    g = gradient(x) do v
        @checkpoint begin
            sum(v)
        end
    end
    @test g[1].data ≈ [1.0, 1.0, 1.0]
end
```

**Step 2: Add include to runtests.jl**

Add to `test/runtests.jl`:
```julia
include("test_integration.jl")
```

**Step 3: Run all tests**

```bash
julia --project=. test/runtests.jl
```
Expected: All tests PASS, including integration tests.

**Step 4: Commit**

```bash
git add test/test_integration.jl test/runtests.jl
git commit -m "test: add integration test for @checkpoint CPU offload simulation"
```

---

## Summary

| Task | Deliverable | Key file |
|---|---|---|
| 1 | Package scaffold | `Project.toml`, `src/YAADE.jl` |
| 2 | Core types | `src/tape.jl` |
| 3 | Tracked{T} | `src/tracked.jl` |
| 4 | Tape ops | `src/tape_ops.jl` |
| 5 | Operator overloads | `src/track_call.jl` |
| 6 | @checkpoint macro | `src/checkpoint.jl` |
| 7 | Backward pass | `src/backward.jl` |
| 8 | Public API | `src/api.jl` |
| 9 | Integration test | `test/test_integration.jl` |

## Known Limitations (Out of Scope)

- `is_gpu` / `to_gpu` for real CUDA arrays requires a `CUDA.jl` package extension (`ext/YAADECUDAExt.jl`) — add after core tests pass
- Mutations (`setindex!`) in traced code are not supported
- Functions without an `rrule` silently pass through untracked — add a debug mode to warn about this if needed
