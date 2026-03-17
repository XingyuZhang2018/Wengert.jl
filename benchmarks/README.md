# Wengert.jl Benchmarks

Comparison of **Wengert.jl** (operator-overloading tape) against **Zygote.jl** (source-to-source IR transformation).

## Setup

```
julia --project=benchmarks benchmarks/bench_vs_zygote.jl          # CPU + GPU
julia --project=benchmarks benchmarks/bench_vs_zygote.jl cpu      # CPU only
julia --project=benchmarks benchmarks/bench_vs_zygote.jl gpu      # GPU only
```

The benchmark environment is isolated from the main package. Dependencies:
```
Wengert (local)   BenchmarkTools   Zygote   CUDA (optional)
```

---

## Results

Hardware: **Intel i9 + NVIDIA RTX 4090**, Julia 1.11, Float64 (CPU) / Float32 (GPU).

### CPU — array ops

| Operation | Wengert | Zygote | Winner |
|-----------|---------|--------|--------|
| `sum(v.^2)` n=16 | 2.0 μs | 0.4 μs | Zygote **4.9×** |
| `sum(v.^2)` n=256 | 2.5 μs | 1.3 μs | Zygote **2.0×** |
| `sum(v.^2)` n=4096 | 6.0 μs | 22.3 μs | Wengert **3.7×** |
| `sum(W*x)` H=32 | 3.3 μs | 0.9 μs | Zygote **3.8×** |
| `sum(W*x)` H=128 | 8.2 μs | 5.2 μs | Zygote **1.6×** |
| `sum(W*x)` H=512 | 376.9 μs | 353.5 μs | Zygote **1.1×** |
| 2-layer H=32 | 5.3 μs | 1.0 μs | Zygote **5.1×** |
| 2-layer H=128 | 39.9 μs | 32.6 μs | Zygote **1.2×** |

Zygote wins on small ops where Wengert's per-operation tape overhead dominates. Wengert catches up as compute cost grows.

### CPU — deep matmul chain (gradient w.r.t. x₀, N sequential layers)

| Config | Wengert | Zygote | Winner |
|--------|---------|--------|--------|
| H=16, N=10 | 30.6 μs | 110.5 μs | Wengert **3.6×** |
| H=16, N=50 | 139.0 μs | 591.5 μs | Wengert **4.3×** |
| H=32, N=20 | 82.4 μs | 244.1 μs | Wengert **3.0×** |
| H=64, N=10 | 86.7 μs | 167.8 μs | Wengert **1.9×** |

Wengert's tape scales better with depth. Zygote's IR overhead grows with the number of operations traced.

### GPU — single matmul gradient (RTX 4090, Float32)

| Config | Wengert | Zygote | Winner |
|--------|---------|--------|--------|
| `sum(W*x)` H=256 | 768 μs | 765 μs | ~equal |
| `sum(W*x)` H=1024 | 134 μs | 105 μs | Zygote **1.3×** |
| `sum(W*x)` H=4096 | 322 μs | 291 μs | Zygote **1.1×** |

On GPU, both are compute-bound for large H; the AD overhead is negligible relative to the matmul.

### GPU — deep chain with checkpointing (H=512, N=20, Float32)

| Strategy | Time | vs plain |
|----------|------|----------|
| Wengert plain | 1118 μs | — |
| Zygote plain | 1187 μs | — |
| Wengert `@checkpoint :recompute` | 1733 μs | +55% |
| Zygote `checkpointed()` | 1618 μs | +36% |

On GPU, Wengert and Zygote are essentially equivalent without checkpointing (~1.06× in Wengert's favour). Zygote's `checkpointed()` has lower recompute overhead (+36%) than Wengert's `:recompute` (+55%) for this workload (large matmuls where compute cost is non-trivial).

---

## Key Takeaways

| Scenario | Recommendation |
|----------|---------------|
| Few large ops (single matmul, GPU) | Either — performance is comparable |
| Deep chains, many layers (CPU) | **Wengert** — up to 4× faster |
| Small ops, shallow graphs (CPU) | **Zygote** — lower fixed overhead |
| GPU memory-constrained models | **Wengert** `@checkpoint :recompute` — zero activation memory |
| GPU, recompute checkpoint overhead matters | Zygote `checkpointed()` has slightly lower overhead for large matmuls |

The core advantage of Wengert is not raw speed but **GPU memory control**: `@checkpoint :recompute` and `@checkpoint` (CPU offload) let you trade compute or PCIe bandwidth for VRAM, which Zygote's source-to-source approach cannot do.
