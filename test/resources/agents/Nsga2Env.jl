
using PyCall
using Random: rand

@pyimport gymnasium.spaces as spaces
@pyimport numpy

mutable struct Environment
    pyenv::PyObject

    events_length::Int

    function Environment(pyenv::PyObject)
        env = new(pyenv)
        env.events_length = 60

        env.pyenv."observation_space" = pycall(
            spaces.Box,
            PyObject,
            0,
            PyObject(numpy.inf);
            shape = (60,),
            dtype = PyObject(numpy.float32),
        )
        env.pyenv."action_space" = pycall(
            spaces.Dict,
            PyObject,
            Dict(
                "events" => pycall(spaces.Discrete, PyObject, env.events_length),
                "variables" =>
                    pycall(spaces.MultiDiscrete, PyObject, fill(7, env.events_length)),
            ),
        )
        return env
    end
end

function step!(env::Environment, actions)
    nactions = size(actions)[1]

    observations = zeros(Float64, (nactions, env.pyenv."observation_space"."shape"[1]))
    rewards = rand(Float64, (nactions, 1))
    terminated = trues(nactions)
    truncated = falses(nactions)
    infos = [Dict{String,Any}() for _ = 1:nactions]

    return observations, rewards, terminated, truncated, infos
end

function reset!(
    env::Environment,
    seed::Union{Int,Nothing} = nothing,
    options::Union{Dict{String,Any},Nothing} = nothing,
)
    observations = zeros(Float64, env.pyenv."observation_space"."shape"[1])
    return observations, Dict{String,Any}()
end

function close!(env::Environment)
    nothing
end

function render(env::Environment, mode)
    nothing
end

function first_update!(action)
    nothing
end

function update!(action)
    nothing
end
