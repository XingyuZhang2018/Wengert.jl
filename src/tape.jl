mutable struct TapeEntry
    pullback::Any
    input_slots::Vector{Int}
    output_slot::Int
end

mutable struct Tape
    entries::Vector{TapeEntry}
    slots::Vector{Any}
    slot_device::Vector{Symbol}  # :gpu | :cpu
    grad_accum::Dict{Int,Any}
    checkpoint_mode::Bool
end
