using ChainRulesCore

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
