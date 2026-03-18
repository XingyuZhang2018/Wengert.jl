using ChainRulesCore

macro checkpoint(args...)
    if length(args) == 1
        # ── Existing CPU-offload behaviour (unchanged) ──────────────────────
        expr = args[1]
        return quote
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

    elseif length(args) == 3 && args[1] == QuoteNode(:recompute)
        # ── Recompute mode ──────────────────────────────────────────────────
        # args = (:recompute, (var1, var2, ...), body_expr)
        _, input_tuple, body = args

        # Extract variable symbols from the input tuple expression
        input_vars = if Meta.isexpr(input_tuple, :tuple)
            input_tuple.args
        else
            [input_tuple]
        end

        # Build anonymous function: (esc(w), esc(xv)) -> esc(body)
        # Using esc on both params and body ensures they refer to the same caller-scope symbols.
        f_expr = Expr(:->, Expr(:tuple, map(esc, input_vars)...), esc(body))

        return quote
            local _tape = current_tape()
            _tape === nothing && error("@checkpoint used outside of pullback/gradient context")
            _recompute_segment(_tape, $f_expr, $(map(esc, input_vars)...))
        end

    else
        error("@checkpoint: use `@checkpoint expr` (CPU offload) or " *
              "`@checkpoint :recompute (vars...) expr` (recompute)")
    end
end

function _recompute_segment(tape::Tape, f::Function, tracked_inputs::AnyTracked...)
    raw_inputs  = map(t -> t.value, tracked_inputs)
    input_slots = Int[t.slot for t in tracked_inputs]

    # ── Forward: run f on a temporary sub-tape ─────────────────────────────
    sub = Tape(TapeEntry[], Any[], Symbol[], Dict{Int,Any}(), false)
    sub_tracked = map(raw_inputs) do v
        s = push_slot!(sub, v)
        make_tracked(v, s, sub)
    end
    result = with_tape(sub) do
        f(sub_tracked...)
    end
    result isa AnyTracked || error(
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
            make_tracked(v, sl, fresh)
        end
        fresh_result = with_tape(fresh) do
            f(fresh_tracked...)
        end
        fresh_result isa AnyTracked || error(
            "_recompute_segment: block returned an untracked value on recompute re-run")
        backward!(fresh, fresh_result.slot, ȳ)
        grads = map(ft -> get(fresh.grad_accum, ft.slot, ZeroTangent()), fresh_tracked)
        return (ChainRulesCore.NoTangent(), grads...)
    end

    push!(tape.entries, TapeEntry(recompute_pb, input_slots, out_slot))
    return make_tracked(y, out_slot, tape)
end
