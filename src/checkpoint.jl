using ChainRulesCore

macro checkpoint(expr)
    quote
        local _tape = current_tape()
        if _tape === nothing
            error("@checkpoint used outside of pullback/gradient context")
        end
        local _prev = _tape.checkpoint_mode
        _tape.checkpoint_mode = true
        try
            $(esc(expr))
        finally
            _tape.checkpoint_mode = _prev
        end
    end
end

function _recompute_segment(tape::Tape, f::Function, tracked_inputs::Tracked...)
    raw_inputs  = map(t -> t.value, tracked_inputs)
    input_slots = Int[t.slot for t in tracked_inputs]

    # ── Forward: run f on a temporary sub-tape ─────────────────────────────
    sub = Tape(TapeEntry[], Any[], Symbol[], Dict{Int,Any}(), false)
    sub_tracked = map(raw_inputs) do v
        s = push_slot!(sub, v)
        Tracked(v, s, sub)
    end
    result = with_tape(sub) do
        f(sub_tracked...)
    end
    result isa Tracked || error(
        "@checkpoint :recompute block must return a tracked value (did it contain any rrule-covered op?)")
    y = result.value

    # Only the output goes on the main tape; sub is discarded here.
    out_slot = push_slot!(tape, y)

    # ── Recompute pullback ──────────────────────────────────────────────────
    function recompute_pb(ȳ)
        fresh = Tape(TapeEntry[], Any[], Symbol[], Dict{Int,Any}(), false)
        fresh_tracked = map(input_slots) do s
            v  = tape.slots[s]
            sl = push_slot!(fresh, v)
            Tracked(v, sl, fresh)
        end
        fresh_result = with_tape(fresh) do
            f(fresh_tracked...)
        end
        backward!(fresh, fresh_result.slot, ȳ)
        grads = map(ft -> get(fresh.grad_accum, ft.slot, nothing), fresh_tracked)
        return (ChainRulesCore.NoTangent(), grads...)
    end

    push!(tape.entries, TapeEntry(recompute_pb, input_slots, out_slot))
    return Tracked(y, out_slot, tape)
end
