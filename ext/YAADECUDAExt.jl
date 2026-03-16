module YAADECUDAExt

using YAADE
using CUDA

# CuArray is a GPU array — offload to CPU during @checkpoint
YAADE.is_gpu(x::CuArray) = true

# Restore a plain Array back to GPU
YAADE.to_gpu(x::Array) = CuArray(x)

end
