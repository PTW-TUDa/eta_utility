module ju_extensions

export Agents, set_logger

include("./utils.jl")

# exports functionality from the agents module
module Agents
    include("etax/agents/NSGA2.jl")
end

end # module
