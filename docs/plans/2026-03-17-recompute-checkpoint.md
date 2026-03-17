# Recompute Checkpoint Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Extend `@checkpoint` to support `:recompute` mode — a segment-level gradient checkpoint that discards intermediate activations and re-runs the forward pass during backward.

**Architecture:** The `@checkpoint :recompute (vars...) begin body end` macro transforms the block into a function and calls `_recompute_segment`, which runs the block on a sub-tape (discarding intermediates), stores only the output on the main tape, and installs a pullback that re-runs the block on a fresh tape during backward. Nested recompute works automatically because `with_tape(fresh)` makes `current_tape()` return the fresh tape inside any re-run.

**Tech Stack:** Julia macros, ChainRulesCore, existing `Tape`/`TapeEntry`/`Tracked` types, `with_tape`, `backward!`.

---

## Context

Read these files before starting — don't modify them, just understand them:

- `src/tape.jl` — `Tape`, `TapeEntry` structs
- `src/tape_ops.jl` — `push_slot!`, `with_tape`, `current_tape`
- `src/checkpoint.jl` — existing `@checkpoint` macro (CPU offload)
- `src/backward.jl` — `backward!`, `accumulate!`
- `src/api.jl` — `pullback`, `gradient`, `_wrap_for_tracking`
- `test/test_checkpoint.jl` — existing checkpoint tests (understand the helpers)

---

### Task 1: Test — `_recompute_segment` basic correctness (CPU)

**Files:**
- Modify: `test/test_checkpoint.jl`

The test should verify that `_recompute_segment` on a simple `W * x` segment produces the
same gradient as running `W * x` directly (no checkpoint).

**Step 1: Add the failing test at the end of `test/test_checkpoint.jl`**

```julia
@testset "_recompute_segment: gradient matches baseline" begin
    include(joinpath(@__DIR__, "..", "src", "api.jl"))

    W = [1.0 2.0; 3.0 4.0]
    x = [0.5, 1.0]

    # Baseline: no checkpoint
    g_plain = gradient(W, x) do w, xv
        sum(w * xv)
    end

    # With recompute checkpoint
    g_ckpt = gradient(W, x) do w, xv
        y = _recompute_segment(current_tape(), (tw, tx) -> sum(tw * tx), w, xv)
        y
    end

    @test g_ckpt[1] ≈ g_plain[1]
    @test g_ckpt[2] ≈ g_plain[2]
end
```

**Step 2: Run to confirm it fails**

```
cd test
julia --project=.. -e 'include("test_checkpoint.jl")'
```

Expected: `UndefVarError: _recompute_segment not defined`

---

### Task 2: Implement `_recompute_segment`

**Files:**
- Modify: `src/checkpoint.jl`

Add after the existing `@checkpoint` macro:

```julia
function _recompute_segment(tape::Tape, f::Function, tracked_inputs::Tracked...)
    raw_inputs  = map(t -> t.value, tracked_inputs)
    input_slots = Int[t.slot for t in tracked_inputs]

    # ── Forward: run f on a temporary sub-tape ─────────────────────────────
    sub = Tape(TapeEntry[], Any[], Symbol[], Dict{Int,Any}(), false)
    sub_tracked = map(raw_inputs) do v
        s = push_slot!(sub, v)
        Tracked(v, s, sub)
    end
    result = with_tape(sub) do
        f(sub_tracked...)
    end
    result isa Tracked || error(
        "@checkpoint :recompute block must return a tracked value (did it contain any rrule-covered op?)")
    y = result.value

    # Only the output goes on the main tape; sub is discarded here.
    out_slot = push_slot!(tape, y)

    # ── Recompute pullback ──────────────────────────────────────────────────
    function recompute_pb(ȳ)
        fresh = Tape(TapeEntry[], Any[], Symbol[], Dict{Int,Any}(), false)
        fresh_tracked = map(input_slots) do s
            v  = tape.slots[s]
            sl = push_slot!(fresh, v)
            Tracked(v, sl, fresh)
        end
        fresh_result = with_tape(fresh) do
            f(fresh_tracked...)
        end
        backward!(fresh, fresh_result.slot, ȳ)
        grads = map(ft -> get(fresh.grad_accum, ft.slot, nothing), fresh_tracked)
        return (ChainRulesCore.NoTangent(), grads...)
    end

    push!(tape.entries, TapeEntry(recompute_pb, input_slots, out_slot))
    return Tracked(y, out_slot, tape)
end
```

Note: `ChainRulesCore` is already `using`'d at the top of the file via the existing macro —
if it isn't, add `using ChainRulesCore` at the top of `checkpoint.jl`.

**Step 3: Re-run Task 1 test**

```
julia --project=.. -e 'include("test_checkpoint.jl")'
```

Expected: all tests pass including the new one.

**Step 4: Commit**

```bash
git add src/checkpoint.jl test/test_checkpoint.jl
git commit -m "feat: add _recompute_segment for recompute checkpointing"
```

---

### Task 3: Test — `@checkpoint :recompute` macro

**Files:**
- Modify: `test/test_checkpoint.jl`

These tests verify the macro syntax works and produces correct gradients.

**Step 1: Add tests**

```julia
@testset "@checkpoint :recompute: single-expression block" begin
    W = [1.0 2.0; 3.0 4.0]
    x = [0.5, 1.0]

    g_plain = gradient(W, x) do w, xv
        sum(w * xv)
    end

    g_ckpt = gradient(W, x) do w, xv
        h = @checkpoint :recompute (w, xv) begin
            w * xv
        end
        sum(h)
    end

    @test g_ckpt[1] ≈ g_plain[1]
    @test g_ckpt[2] ≈ g_plain[2]
end

@testset "@checkpoint :recompute: multi-step block" begin
    W = [1.0 2.0; 3.0 4.0]
    x = [0.5, 1.0]

    g_plain = gradient(W, x) do w, xv
        tmp = w * xv
        sum(tmp .^ 2)
    end

    g_ckpt = gradient(W, x) do w, xv
        out = @checkpoint :recompute (w, xv) begin
            tmp = w * xv
            tmp .^ 2
        end
        sum(out)
    end

    @test g_ckpt[1] ≈ g_plain[1]
    @test g_ckpt[2] ≈ g_plain[2]
end

@testset "@checkpoint :recompute: error outside tape context" begin
    @test_throws ErrorException @checkpoint :recompute (x,) begin
        x .^ 2
    end
end
```

**Step 2: Run to confirm they fail**

```
julia --project=.. -e 'include("test_checkpoint.jl")'
```

Expected: macro syntax error or `UndefVarError`.

---

### Task 4: Implement `@checkpoint :recompute` macro extension

**Files:**
- Modify: `src/checkpoint.jl`

Replace the existing `macro checkpoint(expr)` with a variadic form that dispatches on argument count:

```julia
macro checkpoint(args...)
    if length(args) == 1
        # ── Existing CPU-offload behaviour (unchanged) ──────────────────────
        expr = args[1]
        return quote
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

    elseif length(args) == 3 && args[1] == QuoteNode(:recompute)
        # ── Recompute mode ──────────────────────────────────────────────────
        # args = (:recompute, (var1, var2, ...), body_expr)
        _, input_tuple, body = args

        # Extract variable symbols from the input tuple expression
        input_vars = if Meta.isexpr(input_tuple, :tuple)
            input_tuple.args          # e.g. [:W, :x]
        else
            [input_tuple]             # single variable
        end

        # Fresh gensym'd parameter names for the anonymous function
        param_syms = [gensym(string("ck_", v)) for v in input_vars]

        # Build the let-rebinding block: let W = p1, x = p2; body end
        let_bindings = length(param_syms) == 1 ?
            Expr(:(=), input_vars[1], param_syms[1]) :
            Expr(:block, [Expr(:(=), v, p) for (v, p) in zip(input_vars, param_syms)]...)

        # Anonymous function: (p1, p2) -> let W=p1, x=p2; body end
        f_expr = Expr(:->, Expr(:tuple, param_syms...),
                      Expr(:let, let_bindings, esc(body)))

        return quote
            local _tape = current_tape()
            _tape === nothing && error("@checkpoint used outside of pullback/gradient context")
            _recompute_segment(_tape, $f_expr, $(map(esc, input_vars)...))
        end

    else
        error("@checkpoint: use `@checkpoint expr` (CPU offload) or " *
              "`@checkpoint :recompute (vars...) expr` (recompute)")
    end
end
```

**Step 3: Re-run all checkpoint tests**

```
julia --project=.. -e 'include("test_checkpoint.jl")'
```

Expected: all tests pass.

**Step 4: Run the full test suite**

```
julia --project=. -e 'using Pkg; Pkg.test()'
```

Expected: all tests pass. If a test fails, read the error before touching anything.

**Step 5: Commit**

```bash
git add src/checkpoint.jl test/test_checkpoint.jl
git commit -m "feat: extend @checkpoint to support :recompute mode"
```

---

### Task 5: Test — nested recompute

**Files:**
- Modify: `test/test_checkpoint.jl`

**Step 1: Add nested test**

```julia
@testset "@checkpoint :recompute: nested checkpoints" begin
    W1 = [1.0 0.5; 0.5 1.0]
    W2 = [2.0 1.0; 1.0 2.0]
    x  = [1.0, 2.0]

    # Baseline: no checkpointing
    g_plain = gradient(W1, W2, x) do w1, w2, xv
        h = w1 * xv
        out = w2 * h
        sum(out)
    end

    # Nested: outer checkpoint wraps inner checkpoint
    g_nested = gradient(W1, W2, x) do w1, w2, xv
        h = @checkpoint :recompute (w1, xv) begin
            w1 * xv
        end
        out = @checkpoint :recompute (w2, h) begin
            w2 * h
        end
        sum(out)
    end

    @test g_nested[1] ≈ g_plain[1]
    @test g_nested[2] ≈ g_plain[2]
    @test g_nested[3] ≈ g_plain[3]
end
```

**Step 2: Run**

```
julia --project=.. -e 'include("test_checkpoint.jl")'
```

Expected: passes without changes (the design already supports this). If it fails, investigate before implementing anything.

**Step 3: Commit (tests only)**

```bash
git add test/test_checkpoint.jl
git commit -m "test: add nested @checkpoint :recompute test"
```

---

### Task 6: GPU integration test

**Files:**
- Modify: `test/test_gpu_checkpoint.jl`

**Step 1: Add test at the end of `test_gpu_checkpoint.jl`**

```julia
@testset "@checkpoint :recompute on GPU: gradients match baseline" begin
    W1 = CUDA.randn(Float32, 4, 4)
    W2 = CUDA.randn(Float32, 4, 4)
    x  = CUDA.randn(Float32, 4)
    y  = CUDA.randn(Float32, 4)

    g_plain = gradient(W1, W2) do w1, w2
        h  = w1 * x
        ŷ  = w2 * h
        d  = ŷ .- y
        sum(d .^ 2)
    end

    g_recompute = gradient(W1, W2) do w1, w2
        h = @checkpoint :recompute (w1,) begin
            w1 * x
        end
        ŷ  = w2 * h
        d  = ŷ .- y
        sum(d .^ 2)
    end

    @test Array(g_recompute[1]) ≈ Array(g_plain[1]) rtol=1e-4
    @test Array(g_recompute[2]) ≈ Array(g_plain[2]) rtol=1e-4
    println("  Max abs diff ∂W1 (recompute vs plain): ",
            maximum(abs, Array(g_plain[1]) .- Array(g_recompute[1])))
end
```

**Step 2: Run GPU test (requires CUDA GPU)**

```
julia --project=. test/test_gpu_checkpoint.jl
```

Expected: all tests pass. This test is skipped automatically on machines without CUDA.

**Step 3: Commit**

```bash
git add test/test_gpu_checkpoint.jl
git commit -m "test: GPU integration test for @checkpoint :recompute"
```

---

### Task 7: Final — full test suite + PR-ready commit

**Step 1: Run full test suite**

```
julia --project=. -e 'using Pkg; Pkg.test()'
```

Expected: all tests pass.

**Step 2: Verify `src/Wengert.jl` exports**

Open `src/Wengert.jl` and confirm `_recompute_segment` is **not** exported (it's internal).
The `@checkpoint` macro is already exported via the module — no changes needed.

**Step 3: Final commit if any cleanup was needed**

```bash
git add -p   # review hunks
git commit -m "chore: cleanup after recompute checkpoint implementation"
```

---

## What was NOT done (YAGNI)

- No `:cpu` explicit mode tag for the existing `@checkpoint` — the single-arg form is unchanged and already does CPU offload.
- No multi-return-value blocks — single last-expression return is sufficient.
- No export of `_recompute_segment` — it's an implementation detail.
- No changes to `Tape`, `TapeEntry`, `backward!`, or `api.jl`.
