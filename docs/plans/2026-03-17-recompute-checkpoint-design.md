# Recompute Checkpoint Design

**Date:** 2026-03-17
**Status:** Approved

---

## Motivation

The existing `@checkpoint` macro implements a **CPU offload** strategy: GPU activations are
copied to CPU during the forward pass and restored to GPU during the backward pass. This
saves GPU VRAM but still consumes CPU RAM proportional to the activation size.

A complementary **recompute** strategy discards intermediate activations entirely during the
forward pass and re-runs the forward computation on demand during the backward pass. This
trades compute for memory: the block's operations are executed twice, but no intermediate
activations are held in memory between forward and backward. This is preferable when:

- the computation is cheap relative to its memory footprint (e.g., element-wise activations
  over large tensors),
- CPU RAM is also scarce, or
- PCIe bandwidth (used by CPU offload) is the bottleneck.

---

## User API

```julia
# Existing (unchanged): CPU offload
@checkpoint begin
    h = W * x
end

# New: recompute checkpoint
h = @checkpoint :recompute (W, x) begin
    tmp = W * x
    relu.(tmp)      # last expression is the return value
end
```

`(W, x)` are the **explicit inputs** to the segment — the `Tracked` values from the outer
scope that the block depends on. The block body may contain arbitrarily many intermediate
operations; none of them appear on the main tape. The last expression is the segment output,
returned as a `Tracked` value on the main tape.

---

## Macro Expansion

```julia
@checkpoint :recompute (W, x) begin
    body
end
```

expands to:

```julia
let _tape = current_tape()
    _tape === nothing && error("@checkpoint used outside of pullback/gradient context")
    local _f = (_ck_arg1, _ck_arg2) -> let W = _ck_arg1, x = _ck_arg2
        body
    end
    _recompute_segment(_tape, _f, W, x)
end
```

The `let W = _ck_arg1, x = _ck_arg2` rebinding isolates the block from the enclosing scope,
ensuring that when the block is re-run during the backward pass the fresh `Tracked` arguments
are used instead of the original ones.

The existing single-argument form `@checkpoint begin ... end` is unchanged.

---

## Core Function: `_recompute_segment`

```julia
function _recompute_segment(tape::Tape, f::Function, tracked_inputs::Tracked...)
    raw_inputs  = map(t -> t.value, tracked_inputs)
    input_slots = [t.slot for t in tracked_inputs]

    # ── Forward: run f on a temporary sub-tape ─────────────────────────────
    sub = Tape(TapeEntry[], Any[], Symbol[], Dict{Int,Any}(), false)
    sub_tracked = map(raw_inputs) do v
        s = push_slot!(sub, v)
        Tracked(v, s, sub)
    end
    result = with_tape(sub) do
        f(sub_tracked...)
    end
    result isa Tracked || error("@checkpoint :recompute block must return a tracked value")
    y = result.value

    # Only the output is pushed onto the main tape; sub is discarded (GC'd).
    out_slot = push_slot!(tape, y)

    # ── Recompute pullback ──────────────────────────────────────────────────
    function recompute_pb(ȳ)
        fresh = Tape(TapeEntry[], Any[], Symbol[], Dict{Int,Any}(), false)
        fresh_tracked = map(input_slots) do s
            v  = tape.slots[s]          # read input value from main tape
            sl = push_slot!(fresh, v)
            Tracked(v, sl, fresh)
        end
        fresh_result = with_tape(fresh) do
            f(fresh_tracked...)         # current_tape() == fresh inside f
        end
        backward!(fresh, fresh_result.slot, ȳ)
        grads = map(ft -> get(fresh.grad_accum, ft.slot, nothing), fresh_tracked)
        return (ChainRulesCore.NoTangent(), grads...)
    end

    push!(tape.entries, TapeEntry(recompute_pb, input_slots, out_slot))
    return Tracked(y, out_slot, tape)
end
```

### Why `tape.slots[input_slots]` is always valid

`input_slots` are slots of the block's inputs on the main tape. These slots are created
before the `@checkpoint :recompute` block is entered (they are the block's inputs, not
its outputs). The backward pass processes entries in reverse order, so by the time
`recompute_pb` is called, the input slots have not yet been freed or overwritten.

---

## Nested Checkpointing

Nested recompute checkpoints are supported transparently. Inside `recompute_pb`, the block
is re-run under `with_tape(fresh)`, so `current_tape()` returns `fresh` during the
re-execution. Any `@checkpoint :recompute` call inside the block will therefore call
`_recompute_segment(fresh, ...)`, creating its own sub-sub-tape. The nesting is unbounded
and behaves identically to Zygote's `checkpoint`:

```julia
Zygote.@adjoint checkpoint(f, args...) =
    f(args...), ȳ -> Zygote._pullback(f, args...)[2](ȳ)
```

Both approaches re-run `f` through the full AD system, so inner checkpoints retain their
semantics during the backward pass.

---

## File Changes

| File | Change |
|------|--------|
| `src/checkpoint.jl` | Extend `@checkpoint` to dispatch on `:recompute` mode; add `_recompute_segment` |
| `test/test_checkpoint.jl` | Unit tests: forward correctness, gradient correctness (vs no-checkpoint baseline), nested recompute |
| `test/test_gpu_checkpoint.jl` | GPU integration test: recompute produces numerically identical gradients to baseline |
| `src/Wengert.jl` | No export changes needed (`_recompute_segment` stays internal) |

**No changes** to `tape.jl`, `tape_ops.jl`, `backward.jl`, or `api.jl`.

---

## Known Limitations

1. **Block must return a `Tracked` value.** If the block contains no tracked operations,
   `_recompute_segment` raises a clear error. This mirrors the behaviour of `pullback`.

2. **Inputs must be `Tracked`.** The variables listed in `(W, x)` must be `Tracked` values
   on the current tape. A descriptive error is emitted if they are not.

3. **Compute cost doubles for the segment.** The block runs once during forward and once
   during backward. This is the fundamental memory/compute tradeoff of recompute
   checkpointing.
