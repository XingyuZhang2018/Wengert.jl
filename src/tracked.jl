struct Tracked{T}
    value::T
    slot::Int
    tape::Tape
end

# AbstractArray interface — only for array-backed Tracked values
Base.size(x::Tracked{<:AbstractArray}) = size(x.value)
Base.getindex(x::Tracked{<:AbstractArray}, i...) = getindex(x.value, i...)
Base.length(x::Tracked{<:AbstractArray}) = length(x.value)
Base.IndexStyle(::Type{<:Tracked{T}}) where {T<:AbstractArray} = IndexStyle(T)
Base.eltype(::Type{Tracked{T}}) where {T<:AbstractArray} = eltype(T)
Base.ndims(::Type{Tracked{T}}) where {T<:AbstractArray} = ndims(T)

# Broadcast support — tell Julia that Tracked is already broadcastable (don't collect it)
Base.broadcastable(x::Tracked{<:AbstractArray}) = x

# iterate is needed by some broadcast fallback paths
Base.iterate(x::Tracked{<:AbstractArray}, state...) = iterate(x.value, state...)
