# YAADE.jl — Reverse-Mode AD Engine Design

**Date:** 2026-03-16
**Status:** Approved

---

## Motivation

Zygote's source-to-source IR transformation causes pullback functions to close over intermediate values at compile time. This makes it structurally impossible to offload activations from GPU to CPU memory during the forward pass (CPU gradient checkpointing). YAADE implements a tape-based reverse-mode AD engine that is compatible with ChainRulesCore rules and supports explicit CPU checkpointing.

---

## Goals

- Reverse-mode AD via a Wengert-list tape
- Full compatibility with ChainRulesCore `rrule` ecosystem
- Explicit `@checkpoint` macro for GPU → CPU activation offloading
- Support for arbitrary struct types (via Functors.jl) as well as arrays and scalars
- Both `pullback` (low-level) and `gradient` (high-level) APIs

## Non-Goals

- Higher-order derivatives
- Forward-mode AD
- Control-flow tracing through arbitrary Julia code (only tracked primitives with `rrule` are traced)

---

## Architecture

### Execution Model

Tape-based (Wengert list). The forward pass records operations sequentially into a flat `Tape`. Each operation that has a ChainRulesCore `rrule` is intercepted: the output value and pullback function are stored as a `TapeEntry`. The backward pass replays the tape in reverse, calling pullbacks and accumulating gradients into `tape.grad_accum`.

### Tracing Mechanism

Operator overloading on `Tracked{T}`. Any `T` is supported; for `T <: AbstractArray` the `AbstractArray` interface is implemented to allow transparent use in existing code. Struct types are handled by `Functors.fmap`, which recursively wraps leaf arrays into `Tracked` values before the forward pass.

---

## Core Types

```julia
mutable struct TapeEntry
    pullback::Any              # pb function returned by ChainRulesCore.rrule
    input_slots::Vector{Int}   # indices into tape.slots for inputs
    output_slot::Int           # index into tape.slots for output
end

mutable struct Tape
    entries::Vector{TapeEntry}
    slots::Vector{Any}          # all intermediate values; index = slot ID
    slot_device::Vector{Symbol} # :gpu | :cpu, parallel to slots
    grad_accum::Dict{Int,Any}   # slot ID → accumulated gradient
    checkpoint_mode::Bool       # true when inside @checkpoint block
end

struct Tracked{T}
    value::T
    slot::Int    # position in tape.slots
    tape::Tape
end

# AbstractArray interface — only for array types
Base.size(x::Tracked{<:AbstractArray})          = size(x.value)
Base.getindex(x::Tracked{<:AbstractArray}, i...) = getindex(x.value, i...)
```

---

## Operator Overloading & ChainRulesCore Integration

When a primitive is called with `Tracked` arguments:

1. Extract raw values from inputs.
2. Call `ChainRulesCore.rrule(f, raw_args...)` → `(y, pb)`.
3. Register `y` in `tape.slots` via `push_slot!` (respects `checkpoint_mode`).
4. Append a `TapeEntry(pb, input_slots, output_slot)` to `tape.entries`.
5. Return `Tracked(y, output_slot, tape)`.

Functions without an `rrule` are not traced; their outputs are not `Tracked`.

```julia
function track_call(f, args::Tracked...)
    raw_args = map(x -> x.value, args)
    y, pb    = ChainRulesCore.rrule(RuleConfig(), f, raw_args...)
    tape     = args[1].tape
    out_slot = push_slot!(tape, y)
    push!(tape.entries, TapeEntry(pb, [x.slot for x in args], out_slot))
    return Tracked(y, out_slot, tape)
end
```

A custom `RuleConfig` subtype is defined so that rules requiring AD-engine capabilities can dispatch correctly:

```julia
struct YAADERuleConfig <: ChainRulesCore.RuleConfig{Union{}} end
```

All common Base operations (`+`, `-`, `*`, `matmul`, etc.) are overloaded for `Tracked` arguments. Mixed `Tracked`/non-`Tracked` calls promote the non-`Tracked` argument to an untracked slot.

Tangent arithmetic uses `ChainRulesCore.add!!` throughout to correctly handle `NoTangent` and `ZeroTangent`.

---

## `@checkpoint` Macro

Marks a code block so that GPU intermediate values produced during the forward pass are immediately offloaded to CPU pinned memory.

```julia
macro checkpoint(expr)
    quote
        _tape      = current_tape()
        _prev      = _tape.checkpoint_mode
        _tape.checkpoint_mode = true
        _result    = $(esc(expr))
        _tape.checkpoint_mode = _prev
        _result
    end
end
```

`push_slot!` checks the flag and offloads if needed:

```julia
function push_slot!(tape::Tape, value)
    push!(tape.slots, value)
    id = length(tape.slots)
    if tape.checkpoint_mode && is_gpu(value)
        tape.slots[id] = Array(value)          # offload to CPU
        push!(tape.slot_device, :cpu)
    else
        push!(tape.slot_device, :gpu)
    end
    return id
end
```

During the backward pass, slots marked `:cpu` are moved back to GPU on demand:

```julia
function get_slot_for_backward(tape, id)
    tape.slot_device[id] === :cpu ? CuArray(tape.slots[id]) : tape.slots[id]
end
```

Nested `@checkpoint` blocks are supported because the flag is saved and restored on entry/exit.

---

## Public API

### `pullback`

```julia
function pullback(f, args...)
    tape = Tape([], [], Symbol[], Dict{Int,Any}(), false)

    tracked_args = map(args) do arg
        Functors.fmap(arg) do leaf
            slot = push_slot!(tape, leaf)
            Tracked(leaf, slot, tape)
        end
    end

    result = with_tape(tape) do
        f(tracked_args...)
    end

    y = untrack(result)

    function back(ȳ)
        backward!(tape, result.slot, ȳ)
        return map(tracked_args) do arg
            Functors.fmap(arg) do leaf_tracked
                get(tape.grad_accum, leaf_tracked.slot, nothing)
            end
        end
    end

    return y, back
end
```

### `gradient`

```julia
function gradient(f, args...)
    y, back = pullback(f, args...)
    return back(one(y))   # ȳ = 1 for scalar outputs
end
```

---

## Backward Pass

```julia
function backward!(tape::Tape, output_slot::Int, ȳ)
    accumulate!(tape, output_slot, ȳ)

    for entry in Iterators.reverse(tape.entries)
        g_out   = tape.grad_accum[entry.output_slot]
        g_inputs = entry.pullback(g_out)   # call ChainRulesCore pb

        for (slot, g) in zip(entry.input_slots, g_inputs)
            accumulate!(tape, slot, g)
        end
    end
end

function accumulate!(tape, slot, grad)
    grad isa ChainRulesCore.NoTangent  && return
    grad isa ChainRulesCore.ZeroTangent && return
    existing = get(tape.grad_accum, slot, nothing)
    tape.grad_accum[slot] = isnothing(existing) ? grad : ChainRulesCore.add!!(existing, grad)
end
```

---

## Data Flow Summary

```
pullback(f, args)
  ├─ Functors.fmap(args) → Tracked leaves registered in tape.slots
  ├─ f(tracked_args)     → forward pass; tape.entries grows
  │    └─ @checkpoint regions → GPU values offloaded to tape.slots as CPU arrays
  └─ back(ȳ)
       ├─ backward!(tape) → reverse tape, GPU↔CPU transfers as needed
       └─ Functors.fmap(args) → extract gradients from grad_accum, restore struct shape
```

---

## Dependencies

| Package | Role |
|---|---|
| `ChainRulesCore.jl` | Rule definitions and tangent types |
| `Functors.jl` | Recursive struct traversal for tracking and gradient extraction |
| `CUDA.jl` | `CuArray` ↔ `Array` transfers for CPU checkpoint |

---

## Out of Scope (Future Work)

- Automatic checkpoint scheduling based on GPU memory budget
- Support for mutation (`setindex!`) in traced code
- Batched / vectorised-map (`vmap`) support
- Higher-order AD
