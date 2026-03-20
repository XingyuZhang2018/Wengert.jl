# test/test_coverage_checkpoint.jl — coverage for checkpoint.jl error paths
using Test
using Wengert

@testset "@checkpoint outside context errors" begin
    @test_throws ErrorException @checkpoint begin
        1 + 1
    end
end

@testset "@checkpoint bad syntax errors" begin
    # Test the macroexpand error for bad argument count
    @test_throws Exception eval(:(@checkpoint :recompute))
end
