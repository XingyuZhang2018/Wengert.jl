# benchmarks/bench_vs_zygote.jl
#
# Wengert.jl vs Zygote.jl — gradient performance comparison
#
# Note: Wengert uses operator-overloading on AbstractArrays.
#       Scalar-only inputs are not supported; all benchmarks use array inputs.
#
# Sections:
#   1. CPU — array ops (broadcast, matmul, various sizes)
#   2. CPU — deep matmul chain (N layers)
#   3. GPU — single matmul gradient
#   4. GPU — deep chain: no checkpoint vs Wengert :recompute vs Zygote.checkpointed
#
# Usage:
#   julia --project=. benchmarks/bench_vs_zygote.jl          # all
#   julia --project=. benchmarks/bench_vs_zygote.jl cpu      # CPU only
#   julia --project=. benchmarks/bench_vs_zygote.jl gpu      # GPU only

using BenchmarkTools, Printf, Statistics
using Wengert
using Zygote

BenchmarkTools.DEFAULT_PARAMETERS.seconds = 10
BenchmarkTools.DEFAULT_PARAMETERS.evals   = 1

SECTION = get(ARGS, 1, "all")

sep(title) = (println(); println("=" ^ 66); println(title); println("=" ^ 66))

function show_row(label, t_w, t_z)
    ratio = t_z / t_w
    tag   = ratio >= 1.0 ? @sprintf("Wengert %.2f× faster", ratio) :
                           @sprintf("Zygote  %.2f× faster", 1/ratio)
    @printf "  %-30s  Wengert %8.1f μs   Zygote %8.1f μs   (%s)\n" label t_w t_z tag
end

# ── Section 1 & 2: CPU ────────────────────────────────────────────────────────

if SECTION in ("all", "cpu")
    sep("SECTION 1 — CPU: array ops")

    # Element-wise broadcast: sum(v .^ 2)
    for n in (16, 256, 4096)
        v = randn(n)
        bw = @benchmark Wengert.gradient(v -> sum(v .^ 2), $v)
        bz = @benchmark Zygote.gradient(v  -> sum(v .^ 2), $v)
        show_row("sum(v.^2) n=$n", median(bw).time/1e3, median(bz).time/1e3)
    end
    println()

    # Matrix-vector: sum(W * x)
    for H in (32, 128, 512)
        W = randn(H, H); x = randn(H)
        bw = @benchmark Wengert.gradient(w -> sum(w * $x), $W)
        bz = @benchmark Zygote.gradient(w  -> sum(w * $x), $W)
        show_row("sum(W*x) H=$H", median(bw).time/1e3, median(bz).time/1e3)
    end
    println()

    # Two-layer: sum(W2 * (W1 * x))
    for H in (32, 128)
        W1 = randn(H, H); W2 = randn(H, H); x = randn(H)
        bw = @benchmark Wengert.gradient($W1, $W2) do w1, w2; sum(w2 * (w1 * $x)); end
        bz = @benchmark Zygote.gradient($W1, $W2)  do w1, w2; sum(w2 * (w1 * $x)); end
        show_row("2-layer H=$H", median(bw).time/1e3, median(bz).time/1e3)
    end

    sep("SECTION 2 — CPU: deep matmul chain (gradient w.r.t. x0)")

    function deep_chain(Ws, x0)
        x = x0
        for W in Ws; x = W * x; end
        sum(x)
    end

    for (H, N) in ((16, 10), (16, 50), (32, 20), (64, 10))
        Ws = [randn(H, H) for _ in 1:N]; x0 = randn(H)
        bw = @benchmark Wengert.gradient(x -> deep_chain($Ws, x), $x0)
        bz = @benchmark Zygote.gradient(x  -> deep_chain($Ws, x), $x0)
        show_row("H=$H N=$N", median(bw).time/1e3, median(bz).time/1e3)
    end
end

# ── Section 3 & 4: GPU ────────────────────────────────────────────────────────

if SECTION in ("all", "gpu")
    using CUDA
    if !CUDA.functional()
        println("\nNo CUDA device — skipping GPU sections")
    else
        println()
        @printf "GPU: %s  (%.1f GB VRAM)\n" CUDA.name(CUDA.device()) CUDA.totalmem(CUDA.device())/2^30
        free_gpu() = (GC.gc(true); CUDA.reclaim())

        # Force CUDA kernel compilation before timing starts
        function gpu_warmup()
            W = CUDA.randn(Float32, 64, 64); x = CUDA.randn(Float32, 64)
            for _ in 1:3
                Wengert.gradient(w -> sum(w * x), W); CUDA.synchronize()
                Zygote.gradient(w  -> sum(w * x), W); CUDA.synchronize()
            end
            free_gpu()
        end
        gpu_warmup()

        sep("SECTION 3 — GPU: single matmul gradient")

        for H in (256, 1024, 4096)
            W = CUDA.randn(Float32, H, H); x = CUDA.randn(Float32, H)
            # Per-size warmup to compile kernels for this shape
            for _ in 1:2
                Wengert.gradient(w -> sum(w * x), W); CUDA.synchronize()
                Zygote.gradient(w  -> sum(w * x), W); CUDA.synchronize()
            end
            free_gpu()
            bw = @benchmark (Wengert.gradient(w -> sum(w * $x), $W); CUDA.synchronize())
            free_gpu()
            bz = @benchmark (Zygote.gradient(w  -> sum(w * $x), $W); CUDA.synchronize())
            free_gpu()
            show_row("sum(W*x) GPU H=$H", median(bw).time/1e3, median(bz).time/1e3)
        end

        sep("SECTION 4 — GPU: deep chain with checkpoint strategies")
        println("  Gradient w.r.t. x0  (H=512, N=20 layers, Float32)")
        println()

        H, N = 512, 20
        Ws = [CUDA.randn(Float32, H, H) ./ sqrt(Float32(H)) for _ in 1:N]
        x0 = CUDA.randn(Float32, H)

        function plain(Ws, x)
            for W in Ws; x = W * x; end; sum(x)
        end
        function wengert_recompute(Ws, x)
            for W in Ws
                x = @checkpoint :recompute (x,) begin W * x end
            end
            sum(x)
        end
        function zygote_ckpt(Ws, x)
            for W in Ws
                x = Zygote.checkpointed((W, x) -> W * x, W, x)
            end
            sum(x)
        end

        free_gpu()
        bw_p = @benchmark (Wengert.gradient(x -> plain($Ws, x),             $x0); CUDA.synchronize())
        free_gpu()
        bz_p = @benchmark (Zygote.gradient(x  -> plain($Ws, x),             $x0); CUDA.synchronize())
        free_gpu()
        bw_r = @benchmark (Wengert.gradient(x -> wengert_recompute($Ws, x), $x0); CUDA.synchronize())
        free_gpu()
        bz_c = @benchmark (Zygote.gradient(x  -> zygote_ckpt($Ws, x),       $x0); CUDA.synchronize())
        free_gpu()

        t_wp = median(bw_p).time / 1e3
        t_zp = median(bz_p).time / 1e3
        t_wr = median(bw_r).time / 1e3
        t_zc = median(bz_c).time / 1e3

        @printf "  %-42s  %8.1f μs\n" "Wengert  plain"                   t_wp
        @printf "  %-42s  %8.1f μs\n" "Zygote   plain"                   t_zp
        @printf "  %-42s  %8.1f μs\n" "Wengert  @checkpoint :recompute"  t_wr
        @printf "  %-42s  %8.1f μs\n" "Zygote   checkpointed()"          t_zc
        println()
        @printf "  Wengert/Zygote plain     : %.2f×  (%s)\n" t_zp/t_wp (t_zp >= t_wp ? "Wengert faster" : "Zygote faster")
        @printf "  Wengert/Zygote checkpoint: %.2f×  (%s)\n" t_zc/t_wr (t_zc >= t_wr ? "Wengert faster" : "Zygote faster")
        @printf "  Wengert recompute overhead: %+.1f%%  vs Wengert plain\n" (t_wr/t_wp - 1)*100
        @printf "  Zygote  checkpoint overhead: %+.1f%%  vs Zygote plain\n"  (t_zc/t_zp - 1)*100
    end
end

println("\nDone.")
