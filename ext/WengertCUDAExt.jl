module WengertCUDAExt

using Wengert
using CUDA

# CuArray is a GPU array — offload to CPU during @checkpoint
Wengert.is_gpu(x::CuArray) = true

# Restore a plain Array back to GPU
Wengert.to_gpu(x::Array) = CuArray(x)

end
