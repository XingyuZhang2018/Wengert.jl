# Strip Tracked wrapper from a value
untrack(x::Tracked) = x.value
untrack(x::TrackedArray) = x.value
untrack(x) = x

# Recursively untrack: walk through @functor structs
function deep_untrack(x)
    if x isa AnyTracked
        return x.value
    end
    children, re = Functors.functor(typeof(x), x)
    if children === x
        return x
    end
    untracked = map(deep_untrack, children)
    try
        return re(untracked)
    catch
        return untracked
    end
end

# ---------------------------------------------------------------------------
# Forward wrapping: walk arg, wrap AbstractArray leaves in TrackedArray.
# For structs with @functor, reconstruct the original struct type with
# TrackedArray fields (TrackedArray <: AbstractArray satisfies type constraints).
# Falls back to NamedTuple if reconstruction fails.
# ---------------------------------------------------------------------------
function _wrap_for_tracking(arg, tape)
    if arg isa AbstractArray
        slot = push_slot!(tape, arg)
        return make_tracked(arg, slot, tape)
    end
    # Check if Functors knows about this type
    children, _re = Functors.functor(typeof(arg), arg)
    if children === arg
        # Not a functor — return as-is
        return arg
    end
    # Struct with @functor: recurse into children, then try to reconstruct
    tracked_children = map(children) do child
        _wrap_for_tracking(child, tape)
    end
    try
        return _re(tracked_children)
    catch
        # Fall back to NamedTuple if reconstruction fails
        return tracked_children
    end
end

# ---------------------------------------------------------------------------
# Backward gradient extraction: walk tracked_arg (same shape as _wrap_for_tracking
# output) and extract gradients. Returns a struct matching the original arg type.
# ---------------------------------------------------------------------------
function _extract_grads(original_arg, tracked_arg, grad_accum)
    if tracked_arg isa AnyTracked
        # Leaf — look up gradient
        return get(grad_accum, tracked_arg.slot, nothing)
    elseif original_arg isa AbstractArray
        # Wrapped array but somehow lost tracking
        return nothing
    end
    # Struct case: tracked_arg may be the reconstructed struct or a NamedTuple
    children, re = Functors.functor(typeof(original_arg), original_arg)
    if children === original_arg
        return nothing
    end
    # Get children of tracked_arg (works for both struct and NamedTuple)
    tracked_children, _ = Functors.functor(typeof(tracked_arg), tracked_arg)
    if tracked_children === tracked_arg
        # tracked_arg is not decomposable — try NamedTuple field access
        tracked_children = tracked_arg
    end
    grad_children = map(fieldnames(typeof(children))) do fname
        orig_child = getfield(children, fname)
        track_child = getfield(tracked_children, fname)
        _extract_grads(orig_child, track_child, grad_accum)
    end
    # Rebuild named tuple with field names
    fnames = fieldnames(typeof(children))
    nt = NamedTuple{fnames}(grad_children)
    # Reconstruct original struct type with gradient values
    try
        return re(nt)
    catch
        return nt
    end
end

function pullback(f, args...)
    tape = Tape(TapeEntry[], Any[], Symbol[], Dict{Int,Any}(), false)

    # Wrap all leaf AbstractArrays in TrackedArray
    # tracked_args may contain TrackedArray leaves or reconstructed structs
    tracked_args = map(arg -> _wrap_for_tracking(arg, tape), args)

    # Run f with tracked inputs, recording operations onto tape
    result = with_tape(tape) do
        f(tracked_args...)
    end

    y = untrack(result)

    function back(ȳ)
        if !(result isa AnyTracked)
            error("pullback: function output was not tracked — ensure it calls operations that have rrule")
        end
        backward!(tape, result.slot, ȳ)

        # Extract gradients, reconstructing original input shapes
        return map(args, tracked_args) do orig_arg, tracked_arg
            _extract_grads(orig_arg, tracked_arg, tape.grad_accum)
        end
    end

    return y, back
end

function gradient(f, args...)
    y, back = pullback(f, args...)
    y isa Number || error("gradient: function must return a scalar, got $(typeof(y))")
    return back(one(y))
end

# ---------------------------------------------------------------------------
# @ignore: evaluate expression but strip Tracked wrappers from the result,
# so the output is treated as a constant by the AD system.
# Equivalent to Zygote.@ignore.
# ---------------------------------------------------------------------------
macro ignore(ex)
    quote
        deep_untrack($(esc(ex)))
    end
end

# ---------------------------------------------------------------------------
# barrier(f, ad_pullback, args...)
#
# Record f(args...) as an atomic operation on the Wengert tape, using an
# external AD backend (e.g. Zygote.pullback) for the backward pass.
#
# - Automatically deep_untracks all args before calling f
# - Wraps outputs: scalars → Tracked, arrays → TrackedArray,
#   @functor structs → reconstruct with TrackedArray fields
# - Records tape entries connecting tracked inputs to tracked outputs
# ---------------------------------------------------------------------------

# Find the tape from any tracked arg (recursing into @functor structs)
function _find_tape_in_args(args)
    for arg in args
        arg isa AnyTracked && return arg.tape
        children, _ = Functors.functor(typeof(arg), arg)
        children === arg && continue
        for child in children
            t = _find_tape_in_args((child,))
            t !== nothing && return t
        end
    end
    return nothing
end

# Collect all tracked leaf slots from args
function _collect_tracked_leaves!(x, leaves::Vector{Int})
    if x isa AnyTracked
        push!(leaves, x.slot)
    else
        children, _ = Functors.functor(typeof(x), x)
        if children !== x
            for child in children
                _collect_tracked_leaves!(child, leaves)
            end
        end
    end
end

# ---------------------------------------------------------------------------
# Wrap barrier output and record on tape.
# zpb: the pullback function from ad_pullback (computed ONCE during forward)
# raw_args: the untracked args (needed to match gradients to input slots)
# ---------------------------------------------------------------------------

# Helper: build a pullback that maps output tangent → input grad leaves
function _make_barrier_pb(zpb, raw_args, tangent_builder)
    return function(ȳ)
        full_tangent = tangent_builder(ȳ)
        raw_grads = zpb(full_tangent)
        # raw_grads[i] = ∂arg_i (ad_pullback returns per-arg grads, no ∂f)
        grad_leaves = Any[]
        for (i, arg) in enumerate(raw_args)
            _collect_grad_leaves!(arg, raw_grads[i], grad_leaves)
        end
        result_tuple = Any[ChainRulesCore.NoTangent()]
        append!(result_tuple, grad_leaves)
        return tuple(result_tuple...)
    end
end

function _wrap_and_record(result, tape, input_slots, zpb, raw_args)
    if result isa Number
        out_slot = push_slot!(tape, result)
        push!(tape.entries, TapeEntry(_make_barrier_pb(zpb, raw_args, identity), input_slots, out_slot))
        return Tracked(result, out_slot, tape)

    elseif result isa AbstractArray
        out_slot = push_slot!(tape, result)
        push!(tape.entries, TapeEntry(_make_barrier_pb(zpb, raw_args, identity), input_slots, out_slot))
        return TrackedArray(result, out_slot, tape)

    elseif result isa Tuple
        wrapped = ntuple(length(result)) do i
            elem = result[i]
            if elem isa Number || elem isa AbstractArray
                _wrap_and_record_leaf(elem, tape, input_slots, zpb, raw_args,
                    ȳ -> _tuple_tangent(result, i, ȳ))
            else
                children, _ = Functors.functor(typeof(elem), elem)
                if children !== elem
                    _wrap_struct_output(elem, tape, input_slots, zpb, raw_args, i, result)
                else
                    elem
                end
            end
        end
        return wrapped
    else
        children, _ = Functors.functor(typeof(result), result)
        if children !== result
            return _wrap_struct_output(result, tape, input_slots, zpb, raw_args, nothing, nothing)
        end
        return result
    end
end

# Scalar/array inside a tuple with a tangent_builder
function _wrap_and_record_leaf(result::Union{Number, AbstractArray}, tape, input_slots,
                               zpb, raw_args, tangent_builder)
    out_slot = push_slot!(tape, result)
    push!(tape.entries, TapeEntry(_make_barrier_pb(zpb, raw_args, tangent_builder), input_slots, out_slot))
    return result isa Number ? Tracked(result, out_slot, tape) : TrackedArray(result, out_slot, tape)
end

# @functor struct output with TrackedArray fields
function _wrap_struct_output(result, tape, input_slots, zpb, raw_args,
                             tuple_idx, full_result)
    T = typeof(result)
    children, re = Functors.functor(T, result)
    combined_slot = push_slot!(tape, result)

    # Main tape entry: Tangent → struct field order tuple → zpb
    function pb_main(∂struct)
        struct_fnames = fieldnames(T)
        ∂tuple = ntuple(length(struct_fnames)) do j
            fn = struct_fnames[j]
            hasproperty(∂struct, fn) ? getproperty(∂struct, fn) : zero(getfield(result, fn))
        end
        full_tangent = if tuple_idx !== nothing
            ntuple(i -> i == tuple_idx ? ∂tuple : nothing, length(full_result))
        else
            ∂tuple
        end
        raw_grads = zpb(full_tangent)
        result_tuple = Any[ChainRulesCore.NoTangent()]
        for (i, arg) in enumerate(raw_args)
            _collect_grad_leaves!(arg, raw_grads[i], result_tuple)
        end
        return tuple(result_tuple...)
    end
    push!(tape.entries, TapeEntry(pb_main, input_slots, combined_slot))

    # Extraction entries: combined → per-field slots
    fnames = fieldnames(typeof(children))
    tracked_children = map(fnames) do fname
        val = getfield(children, fname)
        if val isa AbstractArray
            field_slot = push_slot!(tape, val)
            push!(tape.entries, TapeEntry(
                ∂val -> (ChainRulesCore.NoTangent(),
                         ChainRulesCore.Tangent{T}(; (fname => ∂val,)...)),
                [combined_slot], field_slot
            ))
            TrackedArray(val, field_slot, tape)
        else
            val
        end
    end

    nt = NamedTuple{fnames}(tracked_children)
    try; return re(nt); catch; return nt; end
end

function _tuple_tangent(tup, i, ∂elem)
    ntuple(j -> j == i ? ∂elem : nothing, length(tup))
end

# Collect gradient leaves matching tracked input structure
function _collect_grad_leaves!(arg, grad, leaves::Vector)
    if arg isa AbstractArray
        push!(leaves, grad)
    else
        children, _ = Functors.functor(typeof(arg), arg)
        if children !== arg && grad !== nothing
            grad_children = try
                gc, _ = Functors.functor(typeof(grad), grad)
                gc
            catch
                grad
            end
            for (fname, child) in zip(fieldnames(typeof(children)), children)
                g = try getfield(grad_children, fname) catch; nothing end
                _collect_grad_leaves!(child, g, leaves)
            end
        end
    end
end

function barrier(f, ad_pullback, args...)
    tape = _find_tape_in_args(args)
    tape === nothing && return f(args...)

    input_slots = Int[]
    for arg in args
        _collect_tracked_leaves!(arg, input_slots)
    end

    raw_args = map(deep_untrack, args)

    # Single forward+pullback call — no recomputation in backward
    result, zpb = ad_pullback(f, raw_args...)

    return _wrap_and_record(result, tape, input_slots, zpb, raw_args)
end
