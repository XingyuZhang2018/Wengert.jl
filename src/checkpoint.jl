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
