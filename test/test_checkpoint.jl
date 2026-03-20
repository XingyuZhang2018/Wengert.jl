# test/test_checkpoint.jl
using Test

using ChainRulesCore
using ChainRules
using Functors

include(joinpath(@__DIR__, "..", "src", "tape.jl"))
include(joinpath(@__DIR__, "..", "src", "tracked.jl"))
include(joinpath(@__DIR__, "..", "src", "tape_ops.jl"))
include(joinpath(@__DIR__, "..", "src", "track_call.jl"))
include(joinpath(@__DIR__, "..", "src", "checkpoint.jl"))
include(joinpath(@__DIR__, "..", "src", "backward.jl"))
include(joinpath(@__DIR__, "..", "src", "api.jl"))

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

@testset "_recompute_segment: gradient matches baseline" begin
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

    # Sequential (not truly nested) recompute checkpoints
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

@testset "@checkpoint bad syntax errors" begin
    @test_throws Exception eval(:(@checkpoint :recompute))
end
