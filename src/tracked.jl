# ---------------------------------------------------------------------------
# Tracked{T} — for scalar / non-array values
# ---------------------------------------------------------------------------
struct Tracked{T}
    value::T
    slot::Int
    tape::Tape
end

# ---------------------------------------------------------------------------
# TrackedArray{T,N,A} <: AbstractArray{T,N} — for array values
# Being a proper AbstractArray subtype lets it satisfy typed struct field
# constraints like `CT <: AbstractArray{<:Number, 2}`.
# ---------------------------------------------------------------------------
struct TrackedArray{T, N, A <: AbstractArray{T, N}} <: AbstractArray{T, N}
    value::A
    slot::Int
    tape::Tape
end

# Union for general dispatch (either kind of tracked value)
const AnyTracked = Union{Tracked, TrackedArray}

# Smart constructor: arrays → TrackedArray, everything else → Tracked
make_tracked(value::AbstractArray, slot::Int, tape::Tape) =
    TrackedArray(value, slot, tape)
make_tracked(value, slot::Int, tape::Tape) =
    Tracked(value, slot, tape)

# ---------------------------------------------------------------------------
# AbstractArray interface for TrackedArray
# ---------------------------------------------------------------------------
Base.size(x::TrackedArray) = size(x.value)
Base.getindex(x::TrackedArray, i...) = getindex(x.value, i...)
Base.length(x::TrackedArray) = length(x.value)
Base.IndexStyle(::Type{<:TrackedArray{T,N,A}}) where {T,N,A} = IndexStyle(A)
Base.eltype(::Type{<:TrackedArray{T}}) where {T} = T
Base.ndims(::Type{<:TrackedArray{T,N}}) where {T,N} = N

Base.similar(x::TrackedArray, ::Type{S}, dims::Dims) where {S} = similar(x.value, S, dims)
Base.axes(x::TrackedArray) = axes(x.value)

# strides / unsafe_convert only for DenseArray-backed TrackedArrays
Base.strides(x::TrackedArray{T,N,A}) where {T,N,A<:DenseArray} = strides(x.value)
Base.unsafe_convert(P::Type{Ptr{S}}, x::TrackedArray{T,N,A}) where {S,T,N,A<:DenseArray} =
    unsafe_convert(P, x.value)

# convert — needed when Functors.re() calls a struct constructor
Base.convert(::Type{A}, x::TrackedArray{T,N,A}) where {T,N,A<:AbstractArray{T,N}} = x
Base.convert(::Type{<:AbstractArray{T,N}}, x::TrackedArray{T,N}) where {T,N} = x

# Broadcast support
Base.broadcastable(x::TrackedArray) = x

# iterate — needed by some broadcast fallback paths
Base.iterate(x::TrackedArray, state...) = iterate(x.value, state...)

# show — prevent REPL from printing every element
Base.show(io::IO, x::TrackedArray) = print(io, "TrackedArray(slot=$(x.slot), $(summary(x.value)))")
Base.show(io::IO, ::MIME"text/plain", x::TrackedArray) = show(io, x)
