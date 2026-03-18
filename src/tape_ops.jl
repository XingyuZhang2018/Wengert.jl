# Task-local tape — thread-safe, no global state
const TAPE_KEY = :wengert_active_tape

current_tape() = get(task_local_storage(), TAPE_KEY, nothing)

function with_tape(f::Function, tape::Tape)
    task_local_storage(TAPE_KEY, tape) do
        f()
    end
end

# GPU detection — returns false for plain CPU arrays
# Can be overridden for CuArray by defining is_gpu(x::CuArray) = true
is_gpu(x) = false
is_gpu(x::TrackedArray) = is_gpu(x.value)  # look through TrackedArray wrapper

function push_slot!(tape::Tape, value)
    # Defensive: unwrap TrackedArray if it somehow reaches here
    value = value isa AnyTracked ? value.value : value
    if tape.checkpoint_mode && is_gpu(value)
        cpu_val = Array(value)   # offload to CPU
        push!(tape.slots, cpu_val)
        push!(tape.slot_device, :cpu)
    else
        push!(tape.slots, value)
        push!(tape.slot_device, :gpu)
    end
    return length(tape.slots)
end

function get_slot_for_backward(tape::Tape, id::Int)
    if tape.slot_device[id] === :cpu
        return to_gpu(tape.slots[id])   # restore to GPU
    end
    return tape.slots[id]
end

# to_gpu stub — identity for CPU arrays; overridden for CuArray
to_gpu(x) = x
