using Test

@testset "YAADE" begin
    include("test_tape.jl")
    include("test_tracked.jl")
    include("test_tape_ops.jl")
    include("test_track_call.jl")
    include("test_checkpoint.jl")
    include("test_backward.jl")
    include("test_api.jl")
end
