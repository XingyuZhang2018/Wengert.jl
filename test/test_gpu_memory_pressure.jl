# test/test_gpu_memory_pressure.jl
#
# GPU memory pressure test:
#   Section "oom"    — Large model (H=8192, N=100, ~25 GB tape) → OOM vs checkpoint
#   Section "timing" — Medium model (H=4096, N=20) → 3-way timing comparison (default)
#                      no checkpoint | CPU checkpoint | recompute checkpoint
#
# Usage:
#   julia --project=. test/test_gpu_memory_pressure.jl          # timing only
#   julia --project=. test/test_gpu_memory_pressure.jl oom      # OOM demo (exits after)
#   julia --project=. test/test_gpu_memory_pressure.jl all      # both (OOM first, then timing)
#
# NOTE: the OOM test exhausts the CUDA memory pool; run it in its own process.

using Test, CUDA, Wengert, Statistics, Printf

CUDA.functional() || (println("No CUDA device — skipping"); exit(0))
println("Device : ", CUDA.name(CUDA.device()))
println("VRAM   : ", round(CUDA.totalmem(CUDA.device()) / 2^30, digits=1), " GB")
println()

SECTION = get(ARGS, 1, "timing")   # "oom" | "timing" | "all"

# ── Helpers ───────────────────────────────────────────────────────────────────

# N sequential matrix multiplications: x_{i+1} = W_i * x_i (W_i fixed, x differentiated)
# Tape retains every activation on GPU → N × H × H × 4 bytes GPU memory
function deep_forward(Ws, x0)
    x = x0
    for W in Ws; x = W * x; end
    return sum(x)
end

# Same, but each activation is checkpointed to CPU immediately after computation
function deep_forward_ckpt(Ws, x0)
    x = x0
    for W in Ws; x = @checkpoint W * x; end
    return sum(x)
end

# Same, but each activation is discarded entirely and recomputed during backward
function deep_forward_recompute(Ws, x0)
    x = x0
    for W in Ws
        x = @checkpoint :recompute (x,) begin
            W * x
        end
    end
    return sum(x)
end

function make_model(H, N, T=Float32)
    Ws = [CUDA.randn(T, H, H) ./ sqrt(T(H)) for _ in 1:N]
    x0 = CUDA.randn(T, H, H)
    return Ws, x0
end

free_gpu() = (GC.gc(true); CUDA.reclaim())
norm_cpu(x) = sqrt(sum(abs2, Array(x)))

# ── Section 1: OOM demonstration ─────────────────────────────────────────────

if SECTION in ("oom", "all")
    H, N = 8192, 100
    act_MB  = H * H * 4 / 2^20
    total_GB = act_MB * N / 2^10
    println("=" ^ 62)
    println("SECTION 1 — Large model: OOM without checkpoint vs CPU checkpoint")
    @printf "  Config    : H=%d, N=%d layers\n" H N
    @printf "  Activation: %.0f MB per step\n" act_MB
    @printf "  Tape total: %.1f GB without checkpoint\n" total_GB
    println("=" ^ 62)

    # ── 1a: without checkpoint → should OOM ──────────────────────────────────
    @testset "No checkpoint: OOM expected (tape ≈ $(round(Int,total_GB)) GB)" begin
        free_gpu()
        vram_free = round(CUDA.free_memory() / 2^30, digits=1)
        println("  Free VRAM before: $vram_free GB")

        Ws, x0 = make_model(H, N)
        oom_hit = false
        try
            g = gradient(x0) do x; deep_forward(Ws, x); end
            println("  No OOM — GPU has more free memory than expected")
        catch e
            if e isa CUDA.OutOfGPUMemoryError || e isa CUDA.CuError ||
               occursin("out of memory", lowercase(string(e)))
                oom_hit = true
                println("  ✓ OOM as expected — GPU memory exhausted at 100%")
            else
                rethrow()
            end
        end
        @test oom_hit
    end

    # After OOM the memory pool is exhausted; restart Julia for the checkpoint test.
    if SECTION == "oom"
        println("\nRun with 'all' or 'ckpt' in a fresh process to test checkpoint recovery.")
        exit(0)
    end

    # ── 1b: with @checkpoint → fits in VRAM ──────────────────────────────────
    println("\nStarting fresh process state for checkpoint test...")
    free_gpu()

    @testset "CPU checkpoint: same model fits in VRAM" begin
        Ws, x0 = make_model(H, N)
        vram_before = CUDA.free_memory()
        println("  Free VRAM before: $(round(vram_before/2^30,digits=1)) GB")

        g = gradient(x0) do x; deep_forward_ckpt(Ws, x); end

        vram_after = CUDA.free_memory()
        peak_delta = (vram_before - vram_after) / 2^20
        println("  Free VRAM after : $(round(vram_after/2^30,digits=1)) GB")
        println("  Peak VRAM delta : $(round(peak_delta,digits=0)) MB  (vs $(round(total_GB*1024)) MB without checkpoint)")

        @test g[1] isa CuArray{Float32}
        @test !any(isnan, Array(g[1]))
        @test size(g[1]) == (H, H)
        println("  ✓ Gradient norm : $(round(norm_cpu(g[1]),digits=4))")
        free_gpu()
    end
end

# ── Section 2: Timing comparison ─────────────────────────────────────────────

if SECTION in ("timing", "all")
    H, N = 4096, 20
    NWARM, NRUNS = 1, 5
    act_MB  = H * H * 4 / 2^20
    total_MB = act_MB * N

    println()
    println("=" ^ 62)
    println("SECTION 2 — Timing: no checkpoint vs CPU checkpoint vs recompute")
    @printf "  Config           : H=%d, N=%d layers\n" H N
    @printf "  Activation       : %.0f MB per step\n" act_MB
    @printf "  Tape (no ck)     : %.0f MB on GPU\n" total_MB
    @printf "  Tape (CPU ckpt)  : ~%.0f MB on GPU (transfers to/from CPU)\n" act_MB
    @printf "  Tape (recompute) : ~0 MB stored  (recomputes each step on backward)\n"
    println("=" ^ 62)

    free_gpu()
    Ws, x0 = make_model(H, N)

    function bench(label, f, x; nwarm=NWARM, nruns=NRUNS)
        for _ in 1:nwarm
            g = f(x); CUDA.synchronize(); free_gpu()
        end
        ts = Float64[]
        for _ in 1:nruns
            CUDA.synchronize()
            t = @elapsed begin
                g = f(x)
                CUDA.synchronize()
            end
            push!(ts, t)
            free_gpu()
        end
        μ, σ = mean(ts)*1e3, std(ts)*1e3
        @printf "  %-22s %.1f ± %.1f ms\n" label μ σ
        return μ
    end

    @testset "Timing comparison (H=$H, N=$N, n=$NRUNS runs)" begin
        println()
        t_plain     = bench("no checkpoint",
            x -> gradient(x0) do x; deep_forward(Ws, x); end, x0)
        t_ckpt      = bench("CPU checkpoint",
            x -> gradient(x0) do x; deep_forward_ckpt(Ws, x); end, x0)
        t_recompute = bench("recompute",
            x -> gradient(x0) do x; deep_forward_recompute(Ws, x); end, x0)

        @printf "\n  CPU ckpt overhead   : %+.1f%%  (PCIe transfers: %d offload + %d restore)\n" (t_ckpt/t_plain-1)*100 N N
        @printf "  Recompute overhead  : %+.1f%%  (recomputes %d matmuls on backward)\n" (t_recompute/t_plain-1)*100 N
        println()
        println("  Strategy comparison:")
        println("    no checkpoint  — fast, high GPU memory (O(N) activations on GPU)")
        println("    CPU checkpoint — medium overhead, medium GPU memory (activations on CPU)")
        println("    recompute      — 2× compute, zero activation memory stored")

        # Verify gradient correctness for both checkpoint strategies
        g_plain     = gradient(x0) do x; deep_forward(Ws, x);           end
        g_ckpt      = gradient(x0) do x; deep_forward_ckpt(Ws, x);      end
        g_recompute = gradient(x0) do x; deep_forward_recompute(Ws, x); end

        err_ckpt      = maximum(abs, Array(g_plain[1]) .- Array(g_ckpt[1]))
        err_recompute = maximum(abs, Array(g_plain[1]) .- Array(g_recompute[1]))
        @printf "\n  Max abs error (CPU ckpt)  : %.2e\n" err_ckpt
        @printf "  Max abs error (recompute) : %.2e\n" err_recompute

        @test Array(g_ckpt[1])      ≈ Array(g_plain[1]) rtol=1e-4
        @test Array(g_recompute[1]) ≈ Array(g_plain[1]) rtol=1e-4
        println("  ✓ Both checkpoint strategies match plain gradient (rtol=1e-4)")
        free_gpu()
    end
end

println("\nDone.")
