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
