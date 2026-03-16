using ChainRulesCore: add!!, unthunk

function accumulate!(tape::Tape, slot::Int, grad)
    grad isa NoTangent   && return
    grad isa ZeroTangent && return
    grad = unthunk(grad)   # materialize any lazy InplaceableThunk / Thunk
    existing = get(tape.grad_accum, slot, nothing)
    tape.grad_accum[slot] = isnothing(existing) ? grad : add!!(existing, grad)
end

function backward!(tape::Tape, output_slot::Int, ȳ)
    accumulate!(tape, output_slot, ȳ)

    for entry in Iterators.reverse(tape.entries)
        g_out = get(tape.grad_accum, entry.output_slot, nothing)
        g_out === nothing && continue   # no gradient flows through this entry

        # pb(ȳ) returns (∂f, ∂arg1, ∂arg2, ...) — skip first element (∂f = NoTangent usually)
        raw_grads = entry.pullback(g_out)

        for (slot, g) in zip(entry.input_slots, Iterators.drop(raw_grads, 1))
            accumulate!(tape, slot, g)
        end
    end
end
