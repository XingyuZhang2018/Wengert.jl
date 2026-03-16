using ChainRulesCore
using ChainRulesCore: unthunk

struct YAADERuleConfig <: RuleConfig{Union{}} end

function track_call(f, args::Tracked...)
    tape = args[1].tape
    raw_args = map(a -> a.value, args)

    result = rrule(YAADERuleConfig(), f, raw_args...)
    if result === nothing
        # No rrule — call f directly, return untracked value
        return f(raw_args...)
    end
    y, pb = result

    out_slot = push_slot!(tape, y)
    push!(tape.entries, TapeEntry(pb, Int[a.slot for a in args], out_slot))
    return Tracked(y, out_slot, tape)
end

function _ensure_tracked(x::Tracked, tape)
    return x
end

function _ensure_tracked(x, tape)
    slot = push_slot!(tape, x)
    return Tracked(x, slot, tape)
end

# Overload common Base operations
for op in (:+, :-, :*, :/)
    @eval begin
        Base.$op(a::Tracked, b::Tracked) = track_call($op, a, b)
        Base.$op(a::Tracked, b) = track_call($op, a, _ensure_tracked(b, a.tape))
        Base.$op(a, b::Tracked) = track_call($op, _ensure_tracked(a, b.tape), b)
    end
end

Base.:-(a::Tracked) = track_call(-, a)

# Reduction operations
Base.sum(a::Tracked{<:AbstractArray}) = track_call(sum, a)

# ---------------------------------------------------------------------------
# Broadcast interception
# ---------------------------------------------------------------------------

struct TrackedStyle <: Base.Broadcast.BroadcastStyle end

Base.Broadcast.BroadcastStyle(::Type{<:Tracked{<:AbstractArray}}) = TrackedStyle()
Base.Broadcast.BroadcastStyle(::TrackedStyle, ::Base.Broadcast.BroadcastStyle) = TrackedStyle()
Base.Broadcast.BroadcastStyle(::Base.Broadcast.BroadcastStyle, ::TrackedStyle) = TrackedStyle()

# Collect tracked and untracked args from a Broadcasted expression
function _extract_broadcast_args(bc::Base.Broadcast.Broadcasted)
    return bc.args
end

function _unwrap_broadcast_arg(arg::Tracked)
    return arg.value
end
_unwrap_broadcast_arg(arg::Base.RefValue) = arg[]
function _unwrap_broadcast_arg(arg::Base.Broadcast.Broadcasted{TrackedStyle})
    # Materialize nested Broadcasted (will record on tape via our copy)
    return Base.copy(arg)
end
_unwrap_broadcast_arg(arg) = arg

# Check if an arg is a "scalar" in broadcast (not iterated over)
_is_broadcast_scalar(::Base.RefValue) = true
_is_broadcast_scalar(::Tracked) = false
_is_broadcast_scalar(x) = !(x isa AbstractArray)

# Recursively find the tape from a Broadcasted expression's args
function _find_tape(args)
    for arg in args
        if arg isa Tracked
            return arg.tape
        elseif arg isa Base.Broadcast.Broadcasted
            t = _find_tape(arg.args)
            t !== nothing && return t
        end
    end
    return nothing
end

function Base.copy(bc::Base.Broadcast.Broadcasted{TrackedStyle})
    # Find the tape recursively from any Tracked argument (possibly nested)
    tape = _find_tape(bc.args)
    tape === nothing && error("TrackedStyle broadcast: no Tracked argument found")

    f = bc.f

    # Step 1: "partially unwrap" — dereference RefValue and materialize nested
    # Broadcasted{TrackedStyle} (recording those sub-operations on the tape).
    # After this step, args may contain Tracked objects (from nested bc) or plain values.
    partial_args = map(bc.args) do arg
        if arg isa Base.RefValue
            arg[]   # dereference scalar ref
        elseif arg isa Base.Broadcast.Broadcasted{TrackedStyle}
            Base.copy(arg)  # materialize nested broadcast, returns Tracked
        else
            arg  # Tracked or plain scalar/array — pass through
        end
    end

    # Step 2: identify which partial_args are Tracked and collect their slots
    tracked_idxs = Int[i for (i, a) in enumerate(partial_args) if a isa Tracked]

    # Step 3: fully unwrap to plain values for rrule / computation
    fully_raw = map(a -> a isa Tracked ? a.value : a, partial_args)

    # Step 4: try rrule for the whole broadcasted expression
    result = rrule(YAADERuleConfig(), Base.Broadcast.broadcasted, f, fully_raw...)
    if result !== nothing
        bc_result, pb = result
        y = collect(bc_result)

        input_slots = Int[partial_args[i].slot for i in tracked_idxs]

        function broadcast_pb_chainrules(ȳ)
            raw_grads = pb(ȳ)
            # raw_grads = (∂broadcasted, ∂f, ∂arg1, ∂arg2, ...)
            arg_grads = raw_grads[3:end]
            result_grads = Any[unthunk(arg_grads[i]) for i in tracked_idxs]
            return tuple(NoTangent(), result_grads...)
        end

        out_slot = push_slot!(tape, y)
        push!(tape.entries, TapeEntry(broadcast_pb_chainrules, input_slots, out_slot))
        return Tracked(y, out_slot, tape)
    end

    # Fallback: element-wise rrule
    _broadcast_elementwise(f, partial_args, fully_raw, tracked_idxs, tape)
end

function _broadcast_elementwise(f, partial_args, fully_raw, tracked_idxs, tape)
    # partial_args: may contain Tracked (from prior nested bc) or plain values
    # fully_raw: same length, all Tracked/RefValue stripped to plain values
    # tracked_idxs: positions in partial_args/fully_raw that are Tracked

    # Apply broadcast on fully_raw to get output
    bc_plain = Base.Broadcast.broadcasted(f, fully_raw...)
    y_raw = collect(bc_plain)

    # Test if element-wise rrule exists using first element of each array arg
    test_scalars = map(fully_raw) do a
        a isa AbstractArray ? a[firstindex(a)] : a
    end
    test_result = rrule(YAADERuleConfig(), f, test_scalars...)

    if test_result === nothing
        # No rrule — return untracked (non-differentiable operation)
        return y_raw
    end

    # Differentiate element-wise: one pullback per output element
    elem_pbs = Vector{Any}(undef, length(y_raw))
    y_vals = similar(y_raw)
    for idx in eachindex(y_raw)
        scalar_args = map(fully_raw) do a
            a isa AbstractArray ? a[idx] : a
        end
        y_i, pb_i = rrule(YAADERuleConfig(), f, scalar_args...)
        y_vals[idx] = y_i
        elem_pbs[idx] = pb_i
    end

    input_slots = Int[partial_args[i].slot for i in tracked_idxs]

    function elementwise_pb(ȳ)
        # Gradient accumulators: one per tracked input
        grad_arrays = Vector{Any}(undef, length(tracked_idxs))
        for (k, ti) in enumerate(tracked_idxs)
            fa = fully_raw[ti]
            grad_arrays[k] = fa isa AbstractArray ? zero(fa) : zero(eltype(y_vals))
        end

        ȳ_arr = ȳ isa AbstractArray ? ȳ : fill(ȳ, size(y_vals))
        for idx in eachindex(y_vals)
            raw_grads = elem_pbs[idx](ȳ_arr[idx])
            # raw_grads = (∂f, ∂arg1, ∂arg2, ...) — one grad per arg in fully_raw
            arg_grads = raw_grads[2:end]
            for (k, ti) in enumerate(tracked_idxs)
                g = unthunk(arg_grads[ti])
                fa = fully_raw[ti]
                if fa isa AbstractArray
                    grad_arrays[k][idx] += g isa Number ? g : g[idx]
                else
                    grad_arrays[k] += g isa Number ? g : sum(g)
                end
            end
        end

        result = (NoTangent(),)
        for k in 1:length(tracked_idxs)
            result = (result..., grad_arrays[k])
        end
        return result
    end

    out_slot = push_slot!(tape, y_vals)
    push!(tape.entries, TapeEntry(elementwise_pb, input_slots, out_slot))
    return Tracked(y_vals, out_slot, tape)
end
