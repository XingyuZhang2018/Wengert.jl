# Wengert.jl

[![CI](https://github.com/XingyuZhang2018/Wengert.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/XingyuZhang2018/Wengert.jl/actions/workflows/CI.yml)
[![GPU](https://github.com/XingyuZhang2018/Wengert.jl/actions/workflows/GPU.yml/badge.svg)](https://github.com/XingyuZhang2018/Wengert.jl/actions/workflows/GPU.yml)
[![Coverage](https://codecov.io/gh/XingyuZhang2018/Wengert.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/XingyuZhang2018/Wengert.jl)

A tape-based reverse-mode automatic differentiation engine for Julia, with native support for **CPU gradient checkpointing** and **recompute checkpointing** on GPU arrays.

Named after Robert Edwin Wengert, who introduced the [Wengert list](https://dl.acm.org/doi/10.1145/355586.364791) (1964) — the foundational data structure used by this package.

## Motivation

[Zygote.jl](https://github.com/FluxML/Zygote.jl) uses source-to-source IR transformation: pullback functions close over intermediate activations **at compile time**, making it impossible to redirect them to CPU memory. For large GPU models where activations exceed VRAM, Zygote simply runs out of memory with no escape hatch.

Wengert.jl uses **operator-overloading tracing** on a generic `Tracked{T}` wrapper. Every intermediate value is stored in an explicit tape slot, and `@checkpoint` simply stores that slot on CPU (`Array`) instead of GPU (`CuArray`). During the backward pass, the value is restored to GPU on demand. No compiler magic required.

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/XingyuZhang2018/Wengert.jl")
```

Requires Julia ≥ 1.9. GPU support requires [CUDA.jl](https://github.com/JuliaGPU/CUDA.jl) (loaded automatically as a package extension).

## Quick Start

### Scalar gradient

```julia
using Wengert

y, back = pullback(x -> x^2 + sin(x), 1.5)
grads = back(1.0)   # grads[1] = 2x + cos(x) at x=1.5
```

### Array gradient

```julia
W = randn(4, 4)
x = randn(4)

g = gradient(W) do w
    sum(w * x)
end
# g[1] is ∂loss/∂W
```

### GPU with @checkpoint (CPU offload)

```julia
using CUDA

W1 = CUDA.randn(Float32, 1024, 1024)
W2 = CUDA.randn(Float32, 1024, 1024)
x  = CUDA.randn(Float32, 1024)

g = gradient(W1, W2) do w1, w2
    h = @checkpoint w1 * x   # activation stored on CPU, not GPU
    sum(w2 * h)
end
```

The `@checkpoint` macro offloads the activation to CPU RAM on the forward pass and restores it to GPU during backpropagation. This trades PCIe bandwidth for GPU memory.

### GPU with @checkpoint :recompute

```julia
g = gradient(W1, W2) do w1, w2
    h = @checkpoint :recompute (w1, x) begin
        w1 * x          # not stored at all — recomputed during backward
    end
    sum(w2 * h)
end
```

The `:recompute` mode stores **no intermediate activations**. During backpropagation the block is re-executed on a fresh tape to recover the needed values. This trades compute for memory — the block runs twice but zero activation memory is held between forward and backward.

Nested recompute checkpoints work automatically.

## API Reference

### `gradient(f, args...)`

Compute the gradient of scalar-valued `f` with respect to `args`. Returns a tuple of gradients, one per argument.

```julia
g = gradient(x -> sum(x .^ 2), randn(4))
# g[1] ≈ 2x
```

### `pullback(f, args...)`

Returns `(y, back)` where `y = f(args...)` and `back(ȳ)` computes vector-Jacobian products.

```julia
y, back = pullback(x -> x .^ 2, randn(4))
grads = back(ones(4))   # grads[1] = 2x
```

### `@checkpoint expr` — CPU offload

Inside a `gradient` or `pullback` call, offloads the result of `expr` to CPU memory. On GPU arrays (`CuArray`), this performs an `Array(value)` copy on the forward pass and `CuArray(value)` on the backward pass. No-op on CPU arrays.

```julia
h = @checkpoint W * x   # GPU→CPU on forward, CPU→GPU on backward
```

### `@checkpoint :recompute (vars...) block` — recompute

Runs `block` on a temporary sub-tape during the forward pass. Only the block's output is stored on the main tape; all intermediate activations inside the block are discarded. During backpropagation the block is re-executed to recover those intermediates.

```julia
h = @checkpoint :recompute (W, x) begin
    tmp = W * x
    relu.(tmp)      # last expression is the output
end
```

`(W, x)` are the block's inputs — the `Tracked` values from the enclosing scope that the block depends on. The block must return exactly one tracked value (the last expression).

## GPU Checkpointing

Without checkpointing, a tape with N layers of H×H activations requires `N × H × H × sizeof(T)` bytes of GPU memory. For H=8192, N=100 with Float32, that is 25 GB — exceeding a 24 GB GPU.

Two strategies are available:

| Strategy | Activation memory | Extra compute | Best when |
|----------|-------------------|---------------|-----------|
| `@checkpoint expr` (CPU offload) | CPU RAM via PCIe | ~1× (transfer overhead) | CPU RAM is plentiful; compute is expensive |
| `@checkpoint :recompute (vars...) block` | None | ~2× (block reruns) | Memory-critical; ops are cheap (element-wise, small matmuls) |

### Memory scaling

For each checkpointed layer, activation memory is:

| Strategy | GPU memory per layer | Total (N=100, H=8192, Float32) |
|----------|---------------------|-------------------------------|
| No checkpoint | H² × 4 B = 256 MB | **25 GB** (OOM on 24 GB GPU) |
| CPU offload | ~0 MB on GPU (on CPU) | **~0 GB** GPU |
| Recompute | 0 MB | **0 GB** |

### Timing overhead (RTX 4090, H=4096, N=20 layers, Float32)

| Strategy | Wall time | vs. no checkpoint | Source of overhead |
|----------|-----------|-------------------|--------------------|
| No checkpoint | ~174 ms | — | — |
| CPU offload | ~718 ms | **+313%** | N PCIe offloads (fwd) + N restores (bwd) |
| Recompute | ~232 ms | **+33%** | Block recomputed once during backward |

CPU offload transfers each activation over PCIe twice (once to CPU on forward, once back to GPU on backward). For large activations or many layers this dominates. Recompute avoids all transfers and holds zero activation memory, but runs the checkpointed segment twice — once on the forward pass and once on the backward pass to recover intermediates.

**Rule of thumb:** prefer recompute when the checkpointed block is compute-cheap relative to its activation size (e.g. element-wise ops, layer norm, small projections). Prefer CPU offload when the block is compute-expensive and PCIe bandwidth is not a bottleneck.

## ChainRulesCore Compatibility

Wengert.jl dispatches `rrule(WengertRuleConfig(), f, args...)` for every traced operation. Any function with a `ChainRulesCore.rrule` definition works automatically — including all rules from [ChainRules.jl](https://github.com/JuliaDiff/ChainRules.jl).

To add a custom rule:

```julia
using ChainRulesCore

function ChainRulesCore.rrule(::ChainRulesCore.RuleConfig, ::typeof(my_fn), x)
    y = my_fn(x)
    function my_fn_pb(ȳ)
        return NoTangent(), ȳ * my_derivative(x)
    end
    return y, my_fn_pb
end
```

### `barrier(f, ad_pullback, args...)` — external AD boundary

Record `f(args...)` as an atomic operation on Wengert's tape, using an external AD backend for the backward pass. Automatically handles:
- **Deep untracking** all args before calling `f`
- **Wrapping outputs**: scalars, arrays, tuples, and `@functor` structs with `TrackedArray` fields
- **Gradient routing** through struct fields via `ChainRulesCore.Tangent`

```julia
using Wengert, Zygote, Functors

struct Env{CT <: AbstractArray{<:Number, 2}}
    C::CT
end
@functor Env

function my_step(M, env)
    return Env(env.C .+ M), norm(M)
end

# Wengert traces the outer loop; Zygote differentiates each step
g = gradient(M) do m
    env = Env(zeros(3, 3))
    for _ in 1:5
        env, err = barrier(my_step, Zygote.pullback, m, env)
    end
    sum(env.C .^ 2)
end
```

This is the key mechanism for combining Wengert with source-transformation AD systems like Zygote. Wengert handles the outer computation graph (loops, composition), while the barrier delegates differentiation of opaque functions (e.g. those using `@tensor`) to Zygote.

## Struct Support

Structs are differentiable via [Functors.jl](https://github.com/FluxML/Functors.jl). Any struct with a `Functors.@functor` declaration can be passed as an argument.

`TrackedArray <: AbstractArray` means tracked arrays satisfy typed field constraints, so structs with parametric fields like `C::CT where CT <: AbstractArray{<:Number, 2}` work automatically:

```julia
using Functors

struct Linear{W <: AbstractMatrix, B <: AbstractVector}
    W::W
    b::B
end
@functor Linear

model = Linear(randn(Float32, 4, 4), zeros(Float32, 4))
g = gradient(model) do m
    sum(m.W * x .+ m.b)
end
# g[1] is a Linear with gradient arrays
```

## Implementation Notes

The core data structure is the **Wengert list** (tape): a sequence of `TapeEntry` records, each storing a pullback closure, input slot indices, and an output slot index. Forward pass traces the computation graph by operator overloading on `TrackedArray{T,N,A} <: AbstractArray{T,N}` (for arrays) and `Tracked{T}` (for scalars). Backward pass replays the tape in reverse, accumulating gradients via `ChainRulesCore.add!!`.

```
Forward:  x → [op₁] → v₁ → [op₂] → v₂ → ... → loss
Backward: ∂loss → [pb₂] → ∂v₁ → [pb₁] → ∂x
```

R.E. Wengert, *"A simple automatic derivative evaluation program"*, CACM 7(8), 1964.
