# Wengert.jl

A tape-based reverse-mode automatic differentiation engine for Julia, with native support for **CPU gradient checkpointing** on GPU arrays.

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

### GPU with @checkpoint

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

The `@checkpoint` macro offloads the result of its expression to CPU RAM immediately after computation. During backpropagation, it is automatically restored to GPU. This trades PCIe bandwidth for GPU memory.

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

### `@checkpoint expr`

Inside a `gradient` or `pullback` call, offloads the result of `expr` to CPU memory. On GPU arrays (`CuArray`), this performs a `Array(value)` copy on the forward pass and `CuArray(value)` on the backward pass. No-op on CPU arrays.

```julia
h = @checkpoint W * x   # GPU→CPU on forward, CPU→GPU on backward
```

## GPU Checkpointing

Without `@checkpoint`, a tape with N layers of H×H activations requires `N × H × H × sizeof(T)` bytes of GPU memory. For H=8192, N=100 with Float32, that is 25 GB — exceeding a 24 GB GPU.

With `@checkpoint`, only one activation is on GPU at a time (the current layer). Peak GPU memory usage is O(1) in depth.

### Memory comparison (RTX 4090, 24 GB VRAM)

| Config | Without checkpoint | With @checkpoint |
|--------|--------------------|-----------------|
| H=8192, N=100 | ❌ OOM (needs 25 GB) | ✅ Fits |
| H=4096, N=20  | ✅ 174 ms | ✅ 1092 ms (+528%) |

The overhead comes from 40 PCIe transfers per backward pass (20 offloads + 20 restores). For models that would otherwise OOM, this is the only option.

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

## Struct Support

Structs are differentiable via [Functors.jl](https://github.com/FluxML/Functors.jl). Any struct with a `Functors.@functor` declaration can be passed as an argument:

```julia
using Functors

struct Linear
    W::Matrix{Float32}
    b::Vector{Float32}
end
@functor Linear

model = Linear(randn(Float32, 4, 4), zeros(Float32, 4))
g = gradient(model) do m
    sum(m.W * x .+ m.b)
end
# g[1] is a NamedTuple (W=..., b=...)
```

## Implementation Notes

The core data structure is the **Wengert list** (tape): a sequence of `TapeEntry` records, each storing a pullback closure, input slot indices, and an output slot index. Forward pass traces the computation graph by operator overloading on `Tracked{T}`. Backward pass replays the tape in reverse, accumulating gradients via `ChainRulesCore.add!!`.

```
Forward:  x → [op₁] → v₁ → [op₂] → v₂ → ... → loss
Backward: ∂loss → [pb₂] → ∂v₁ → [pb₁] → ∂x
```

R.E. Wengert, *"A simple automatic derivative evaluation program"*, CACM 7(8), 1964.
