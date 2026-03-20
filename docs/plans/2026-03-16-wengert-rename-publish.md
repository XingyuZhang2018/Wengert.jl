# Wengert.jl — Rename & Publish Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Rename the package from YAADE to Wengert, write an English README, and publish to GitHub as a public repository.

**Architecture:** Pure renaming + documentation pass. No logic changes. Four independent tasks: rename core files, update test files, write README, create GitHub repo and push.

**Tech Stack:** Julia package conventions, Git, GitHub REST API (PowerShell `Invoke-RestMethod`), Markdown.

---

### Task 1: Rename Project.toml and extension

**Files:**
- Modify: `Project.toml`
- Rename+modify: `ext/YAADECUDAExt.jl` → `ext/WengertCUDAExt.jl`

**Step 1: Update Project.toml**

Replace the entire file content with:

```toml
name = "Wengert"
uuid = "00000000-0000-0000-0000-000000000001"
authors = ["XingyuZhang2018"]
version = "0.1.0"

[deps]
CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
ChainRules = "082447d4-558c-5d27-93f4-14fc19e9eca2"
ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
Functors = "d9f16b24-f501-4c13-a1f2-28368ffc5196"

[extensions]
WengertCUDAExt = "CUDA"

[compat]
CUDA = "5, 6"
ChainRules = "1.73.0"
ChainRulesCore = "1"
Functors = "0.4"
julia = "1.9"

[extras]
CUDA = "052768ef-5323-57f5-be7d-86a58d49af14"
Test = "8dfed614-e22c-358a-9a83-a03f1d5d13c3"

[targets]
test = ["Test", "CUDA"]
```

**Step 2: Rename and update extension file**

Delete `ext/YAADECUDAExt.jl` and create `ext/WengertCUDAExt.jl` with content:

```julia
module WengertCUDAExt

using Wengert
using CUDA

# CuArray is a GPU array — offload to CPU during @checkpoint
Wengert.is_gpu(x::CuArray) = true

# Restore a plain Array back to GPU
Wengert.to_gpu(x::Array) = CuArray(x)

end
```

**Step 3: Commit**

```bash
git rm ext/YAADECUDAExt.jl
git add Project.toml ext/WengertCUDAExt.jl
git commit -m "refactor: rename package YAADE → Wengert, update extension"
```

---

### Task 2: Rename main module file

**Files:**
- Rename+modify: `src/YAADE.jl` → `src/Wengert.jl`

**Step 1: Delete old file, create new one**

Delete `src/YAADE.jl` and create `src/Wengert.jl` with content:

```julia
module Wengert

using ChainRulesCore
using ChainRules
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

**Step 2: Commit**

```bash
git rm src/YAADE.jl
git add src/Wengert.jl
git commit -m "refactor: rename src/YAADE.jl → src/Wengert.jl"
```

---

### Task 3: Update all test files

**Files:**
- Modify: `test/runtests.jl`
- Modify: `test/test_api.jl`
- Modify: `test/test_integration.jl`
- Modify: `test/test_gpu_checkpoint.jl`
- Modify: `test/test_gpu_memory_pressure.jl`

**Step 1: Update runtests.jl**

Replace:
```julia
using YAADE

@testset "YAADE" begin
```
With:
```julia
using Wengert

@testset "Wengert" begin
```

**Step 2: Update test_api.jl**

Replace `using YAADE` with `using Wengert` (line 3).

**Step 3: Update test_integration.jl**

Replace all occurrences:
- `using YAADE` → `using Wengert`
- `YAADE.is_gpu` → `Wengert.is_gpu`
- `YAADE.to_gpu` → `Wengert.to_gpu`
- `YAADE.Tape` → `Wengert.Tape`
- `YAADE.TapeEntry` → `Wengert.TapeEntry`
- `YAADE.with_tape` → `Wengert.with_tape`
- `YAADE.push_slot!` → `Wengert.push_slot!`

**Step 4: Update test_gpu_checkpoint.jl**

Replace all occurrences:
- `using YAADE` → `using Wengert`
- `YAADE.is_gpu` → `Wengert.is_gpu`
- `YAADE.to_gpu` → `Wengert.to_gpu`

**Step 5: Update test_gpu_memory_pressure.jl**

Replace `using YAADE` → `using Wengert`.

**Step 6: Verify tests pass**

```bash
cd "D:/1 - research/1.9 - AutoDiff/YAADE.jl"
julia --project=. -e "using Pkg; Pkg.test()"
```

Expected: all 55 CPU tests pass (GPU tests are standalone scripts, not included in `Pkg.test()`).

**Step 7: Commit**

```bash
git add test/
git commit -m "refactor: update test files YAADE → Wengert"
```

---

### Task 4: Write README.md

**Files:**
- Create: `README.md`

**Step 1: Create README.md** with the following content:

```markdown
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

Wengert.jl dispatches `rrule(YAADERuleConfig(), f, args...)` for every traced operation. Any function with a `ChainRulesCore.rrule` definition works automatically — including all rules from [ChainRules.jl](https://github.com/JuliaDiff/ChainRules.jl).

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
```

**Step 2: Commit**

```bash
git add README.md
git commit -m "docs: add English README for Wengert.jl"
```

---

### Task 5: Create GitHub repository and push

**Files:** None (Git operations only)

**Step 1: Create repository via GitHub API**

Run in PowerShell:

```powershell
$token = [System.Environment]::GetEnvironmentVariable('GITHUB_TOKEN', 'User')
$headers = @{
    Authorization = "Bearer $token"
    'User-Agent'  = 'Wengert-setup'
    Accept        = 'application/vnd.github+json'
}
$body = @{
    name        = "Wengert.jl"
    description = "Tape-based reverse-mode AD for Julia with CPU gradient checkpointing"
    private     = $false
    auto_init   = $false
} | ConvertTo-Json

$resp = Invoke-RestMethod -Uri 'https://api.github.com/user/repos' -Method POST -Headers $headers -Body $body -ContentType 'application/json'
Write-Host $resp.html_url
```

Expected output: `https://github.com/XingyuZhang2018/Wengert.jl`

**Step 2: Add remote and push**

```bash
cd "D:/1 - research/1.9 - AutoDiff/YAADE.jl"
git remote add origin https://github.com/XingyuZhang2018/Wengert.jl.git
git push -u origin master
```

Expected: all commits pushed, branch tracking set.

**Step 3: Verify**

```bash
git remote -v
git log --oneline origin/master
```

Expected: remote shows `origin`, log matches local.
