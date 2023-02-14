__precompile__()
module Nsga2

import Base: length, iterate, getindex, firstindex, lastindex, eltype, hash
import Random: seed!, shuffle!, rand, AbstractRNG, Xoshiro
import StatsBase
using Logging
using PyCall

export Algorithm, initialize_rnd!, evolve!, evaluate!, create_generation, py_actions, py_store_reward

const np = PyNULL()

__init__() = copy!(np, pyimport("numpy"))

# ----- Types
"""
Parameters for a single variable.
"""
VariableParameter = PyObject

"""
Parameters to describe the properties of many variables.
"""
VariableParameters = Vector{PyObject}

# ----- Solution
"""
Solution object which contains the events and variables used by the algorithm.

:param event_params: Length of the events vector.
:param variable_params: Parameters for the variables vector.
:param max_value: Maximum reward value.
:param empty: Selector to initialize an empty or full events vector.
"""
mutable struct Solution{T}
    events::Vector{Int}
    variables::Vector{T}
    reward::Vector{Float64}
    dominates::Vector{Int}
    dominatedby::Int
    rank::Int
    crowdingdistance::Float64

    function Solution(event_params::Int, variable_params::VariableParameters, max_value, empty::Bool)
        # Create full or empty events
        events = empty ? Vector{Int}(undef, event_params) : collect(UnitRange(0, event_params - 1))

        # Check variables type and create corresponding array
        T = "float" in (var.dtype for var in variable_params) ? Float64 : Int
        variables = Vector{T}(undef, length(variable_params))

        # Check max reward value and create corresponding array
        reward = Float64[max_value]

        new{T}(events, variables, reward, Int[], 0, typemax(Int), 0.0)
    end

    function Solution(events::Vector{Int}, variables, max_value::Float64)
        T = eltype(variables)
        new{T}(events, variables, Float64[max_value], Int[], 0, typemax(Int), 0.0)
    end
end

"""
Create and return the hash of a Solution object.

:param solution: Solution object.
:return: Hash of the solution.
"""
hash(solution::Solution{T}) where {T} = begin
    hash(Actions(solution))
end

"""
Define an Iterator and item getter for the actions array of the Solution object.
"""
struct Actions{T}
    lenvars::UInt
    lenevents::UInt
    count::Int
    eltype::Type{T}
    solution::Solution{T}
    function Actions(solution::Solution{T}) where {T}
        new{T}(
            length(solution.variables),
            length(solution.events),
            length(solution.variables) + length(solution.events),
            T,
            solution,
        )
    end
end

function getindex(actions::Actions{T}, i)::T where {T}
    1 <= i <= actions.count || throw(BoundsError(actions, i))
    if i <= actions.lenevents
        return actions.solution.events[i]
    else
        return actions.solution.variables[i-actions.lenevents]
    end
end

firstindex(actions::Actions) = 1
lastindex(actions::Actions) = actions.count
length(actions::Actions) = lastindex(actions::Actions)
eltype(actions::Actions) = actions.eltype

function iterate(actions::Actions{T})::Union{Tuple{T, Int64}, Nothing} where {T}
    if actions.lenvars == 0 && actions.lenevents == 0
        return nothing
    end
    return actions[1], 2
end

function iterate(actions::Actions{T}, state)::Union{Tuple{T, Int64}, Nothing} where {T}
    if !(firstindex(actions) <= state <= lastindex(actions))
        return nothing
    end
    return actions[state], state + 1
end

"""
Create and return the hash of an Actions object.

:param actions: Actions object.
:return: Hash of the actions.
"""
hash(actions::Actions{T}) where {T} = begin
    hash(collect(actions), zero(UInt64))
end

"""
Create a random variable value depending on the variable parameters.

:param variable_param: Parameters to describe properties of the variable.
:return: Random variable value within boundaries described by parameter.
"""
function randvar(rng::AbstractRNG, variable_param::VariableParameter)
    if variable_param.dtype == "int"
        return rand(rng, variable_param.minimum:(variable_param.maximum-1))
    else
        return rand(rng) * (variable_param.maximum - variable_param.minimum) + variable_param.minimum
    end
end

"""
Randomize both chromosomes of the solution. This essentially reinitializes the solution, but it does
not guarantee that is a new solution.

:param solution: Solution that will be randomized.
:param variable_params: Parameters to describe properties of the variables.
"""
function randomize!(rng::AbstractRNG, solution::Solution, variable_params::VariableParameters)
    @debug "Randomizing solution!"
    # randomize events
    shuffle!(rng, solution.events)

    # randomize variables
    for i in eachindex(solution.variables)
        solution.variables[i] = randvar(rng, variable_params[i])
    end
end

"""
Split the probability between the events and variables. This is done by getting a random
distribution value, which is then used to calculate the mutation rate for each of the two chromosomes.
The function makes sure, that there is always at least one gene being changed, if one of the two
chromosomes is mutated or crossed over according to the random distribution in combination
with the probability.

:param rng: Random generator for generating the probability.
:param nevents: Length of the events chromosomes.
:param nvariables: Length of the variables chromosome.
:param probility: General probility.
"""
function distribute_rates(rng::AbstractRNG, nevents, nvariables, probability)
    # Return probability directly, if one of the chromosomes is missing
    if nevents == 0
        return 0, probability
    elseif nvariables == 0
        return probability, 0
    end

    # Determine a random distribution value between the two chromosomes.
    distribution = rand(rng)

    # Check if either of the two mutation rates would be smaller than one. If it would be, set the distribution,
    # such that each chromosome has at least a single change.
    mutatedevents = (nevents * distribution * probability)
    mutatedvars = (nvariables * (1 - distribution) * probability * 2)
    if distribution <= 0.5 && mutatedevents < 1
        distribution = 1 / (nevents * probability)
    elseif distribution > 0.5 && mutatedvars < 1
        distribution = 1 - (1 / (nvariables * probability * 2))
    end

    rateevents = max(distribution * probability, 0.0)
    ratevariables = max((1 - distribution) * probability * 2, 0.0)
    return rateevents, ratevariables
end
"""
Mutate values of the chromosomes. Returns a new genetic solution and does not modify the current solution.

:param solution: Offspring solution.
:param parent: Parent solution which will be mutated.
:param probability: Mutation probability.
:param variable_params: Paramters for variable mutation.
"""
function mutate!(
    solution::Solution,
    parent::Solution,
    rng::AbstractRNG,
    probability,
    variable_params::VariableParameters,
)
    # Probability can be understood as a fraction of the solution that will be changed
    nevents = length(solution.events)
    nvariables = length(solution.variables)

    rateevents, ratevariables = distribute_rates(rng, nevents, nvariables, probability)

    # Mutate events
    nsamples = floor(Int, nevents * rateevents)
    samples = StatsBase.sample(rng, 1:nevents, ceil(Int, nsamples * 2), replace=false)
    # Copy all values that were not sampled.
    solution.events[:] = parent.events[:]

    # Copy and interchange the sampled values
    for i in 1:nsamples
        solution.events[samples[i]], solution.events[samples[nsamples+i]] =
            parent.events[samples[nsamples+i]], parent.events[samples[i]]
    end

    # Mutate variables
    samples = StatsBase.sample(rng, 1:nvariables, ceil(Int, nvariables * ratevariables), replace=false, ordered=true)
    advance = 1
    for i in eachindex(solution.variables)
        if advance <= length(samples) && i == samples[advance]
            solution.variables[i] = randvar(rng, variable_params[i])
            advance += 1
        else
            solution.variables[i] = parent.variables[i]
        end
    end
end

"""
Cross the solution with another solution. Returns a new genetic solution and does not modify the
current solution.

:param solution: parent envolved in the crossover.
:param other: second parent envolve in the crossover.
:param probability: probability for crossover.
:param maxcrosslen: maximum length of the crossover section.
"""
function crossover!(solution::Solution, other::Solution, rng::AbstractRNG, probability, maxcrosslen)
    nevents = length(solution.events)
    nvariables = length(solution.variables)

    rateevents, ratevariables = distribute_rates(rng, nevents, nvariables, probability)

    # Cross over the events chromosome
    len = floor(Int, min(nevents * rateevents * 2, maxcrosslen))     # length of the crossover section
    start = floor(Int, rand(rng, 1:(nevents-len)))     # starting index of the crossover section

    previous_solutionevents = copy(solution.events)
    # Cross over the selected section
    solution.events[start:start+len-1] = other.events[start:start+len-1]
    # Insert all elements that are not in the crossover section
    idx = 1
    for event in previous_solutionevents
        if start <= idx < start + len
            idx = start + len
        end

        if !(event in other.events[start:start+len-1])
            solution.events[idx] = event
            idx += 1
        end
    end

    # Cross over the variables chromosome
    len = floor(Int, min(nvariables * ratevariables, maxcrosslen))     # length of the crossover section
    start = floor(Int, rand(rng, 1:(nvariables-len)))     # starting index of the crossover section

    solution.variables[start:start+len-1] = other.variables[start:start+len-1]
end

# ------ Generation
Generation = AbstractVector{Solution{T}} where {T}

function py_actions(generation::Generation)
    solution = generation[1]
    var_dtype = typeof(solution.variables[1]) <: AbstractFloat ? np.float64 : np.int64

    dtype = pycall(
        np.dtype,
        PyObject,
        [("events", (np.int64, length(solution.events))), ("variables", (var_dtype, length(solution.variables)))],
    )

    @pycall np.core.records.fromrecords([(s.events, s.variables) for s in generation]; dtype=dtype)::PyObject
end

py_store_reward(generation::Generation, rewards::PyArray) =
    for i in eachindex(generation)
        generation[i].reward::Vector{Float64} = rewards[i, :]
    end

load_generation(events::PyArray, variables::PyArray, max_value::Float64) =
    collect(Solution(events[i, :], variables[i, :], max_value) for i in 1:size(events)[1])

# ----- Algorithm
mutable struct Algorithm
    population::Int
    mutations::Float64
    crossovers::Float64
    maxcrosslen::Float64
    maxretries::Int
    events::Int
    variable_params::VariableParameters
    max_reward::Float64
    sense::String

    seed::UInt64
    rng::Xoshiro

    seensolutions::Vector{UInt64}
    currentminima::Vector{Float64}
    function Algorithm(
        population,
        mutations,
        crossovers,
        maxcrosslen,
        maxretries,
        events,
        variable_params,
        max_reward,
        sense,
        seed,
    )
        rng = Xoshiro()
        seed!(rng, seed)

        new(
            population,
            mutations,
            crossovers,
            maxcrosslen,
            maxretries,
            events,
            variable_params,
            max_reward,
            sense,
            seed,
            rng,
            UInt64[],
            Float64[],
        )
    end
end

"""
Seed the random number generator of the algorithm.

:param elgo: The algorithm.
:param seed: An integer used for seeding the RNG.
"""
seed!(algo::Algorithm, seed) = seed!(algo.rng, seed)

"""
Create a new generation of solutions. The generation can contain all empty solutions
or (partially) initialized solutions.

:param empty: Set this to false if the events chromosome should be initialized with a range of numbers.
:return: A generation of Solution objects.
"""
create_generation(algo::Algorithm, empty) =
    collect(Solution(algo.events, algo.variable_params, algo.max_reward, empty) for _ in 1:algo.population)

"""
Check whether the solution has been seen before (hash is in seen solutions).

:param algo: The algorithm.
:para solution: Solution object to check.
"""
function seensolution(algo::Algorithm, solution::Solution)
    _hash = hash(solution)
    if !(_hash in algo.seensolutions)
        push!(algo.seensolutions, _hash)
        return false
    end

    return true
end

"""
Initialize a generation with random values.

:param algo: The algorithm.
:param generation: The generation with all solutions from python.

:return retries: Number of retries needed to create unique solutions.
"""
function initialize_rnd!(algo::Algorithm, generation::Generation)
    # Make sure that each of the new solutions is unique and has not been seen before.
    @debug "Checking whether any solutions have been seen before."
    retries = 0
    for solution in generation
        randomize!(algo.rng, solution, algo.variable_params)

        while retries <= algo.maxretries
            if !seensolution(algo, solution)
                break
            end
            randomize!(algo.rng, solution, algo.variable_params)
            retries += 1
        end

        if retries >= algo.maxretries
            error("There were too many retries due to equivalent solutions.")
        end
    end
    return retries
end

function initialize_rnd!(algo::Algorithm, generation::Generation, solutions::Vector{Int})
    # Make sure that each of the new solutions is unique and has not been seen before.
    @debug "Checking whether any solutions have been seen before."
    retries = 0
    for s in solutions
        randomize!(algo.rng, generation[s], algo.variable_params)

        while retries <= algo.maxretries
            if !seensolution(algo, generation[s])
                break
            end
            randomize!(algo.rng, generation[s], algo.variable_params)
            retries += 1
        end

        if retries >= algo.maxretries
            error("There were too many retries due to equivalent solutions.")
        end
    end
    return retries
end

"""
Create offspring generation based on parent generation. Apply, crossover, mutations and randomize to
create new solutions.

:param algo: The algorithm.
:param generation: empty offspring generation.
:param generationparent: parent generation.
:param currentlearningrate: current learning rate used to reduce the mutation and crossover rates.

:return retries: Number of retries needed to evolve unique solutions.
"""
function evolve!(algo::Algorithm, generation::Generation, generationparent::Generation, currentlearningrate)
    # Adjust mutation and crossover probability according to learning
    mutations = algo.mutations * currentlearningrate
    crossovers = algo.crossovers * currentlearningrate

    population = length(generation)
    lengenome = length(generation[1].events) + length(generation[1].variables)

    # Perform mutation for the entire generation and store the results in the offspring generation.
    @debug "Mutating generation."
    # Number of solutions to be mutated such that each mutated solution has at least two mutations.
    nsolutions = min(floor(Int, population * lengenome * mutations / 2), population)
    adjustedrate = population * lengenome * mutations / (nsolutions * lengenome)
    mutatesolutions = StatsBase.sample(algo.rng, 1:population, nsolutions, replace=false)

    for i in mutatesolutions
        mutate!(generation[i], generationparent[i], algo.rng, adjustedrate, algo.variable_params)
    end

    # Perform crossover for the entire generation and store the results in the offspring generation.
    @debug "Perfoming crossover for generation."
    matchesfrom = StatsBase.sample(algo.rng, 1:population, ceil(Int, population * crossovers), replace=false)
    matchesto = StatsBase.sample(
        algo.rng,
        [i for i in eachindex(generation) if !(i in matchesfrom)],
        ceil(Int, population * crossovers),
        replace=false,
    )

    adjustedrate = population * lengenome * crossovers / (length(matchesfrom) * lengenome)
    for i in eachindex(matchesfrom)
        crossover!(generation[matchesto[i]], generation[matchesfrom[i]], algo.rng, adjustedrate, algo.maxcrosslen)
    end

    # Make sure that each of the new solutions is unique and has not been seen before.
    @debug "Checking whether any solutions have been seen before."
    retries = 0
    for solution in generation
        while retries <= algo.maxretries
            if !seensolution(algo, solution)
                break
            end
            randomize!(algo.rng, solution, algo.variable_params)
            retries += 1
        end

        if retries >= algo.maxretries
            error("There were too many retries due to equivalent solutions.")
        end
    end
    return retries
end

"""
Return a specific index from the entire population. This handles the generation and generationparent
as if they were concatenated to [generation; generationparent].

:param generation: Main generation.
:param generationparent: Parent generation.
:param idx: Index to fetch.
:return: Value of the concatenated array at the given index.
"""
function population_idx(generation::Generation, generationparent::Generation, idx::Int)
    if idx > length(generation)
        return generationparent[idx-length(generation)]
    else
        return generation[idx]
    end
end

"""
Reset sorting values for all solutions in a generation.

:param generation: Generation to operate over
"""
reset_sorting_values!(generation::Generation) =
    for sol in generation
        sol.dominates = Int64[]
        sol.dominatedby = 0
        sol.rank = typemax(Int)
        sol.crowdingdistance = 0
    end

"""
Evaluate two generations using non-dominated sort and crowding distance sort and return a new
parent generation.

:param algo: The algorithm.
:param generation: Main generation.
:param generationparent: Parent generation.
:param offspring: New generation generated by sorting solution.
:return: Minimum values which were found during the sorting.
:return: List of all seen solutions.
"""
function evaluate!(algo::Algorithm, generation::Generation, generationparent::Generation, offspring::Generation)
    fronts, frontlengths = sort_nondominated(algo, generation, generationparent)

    # Find the last front, which still fits into the population and cut frontlengths off
    for f in eachindex(frontlengths)
        if frontlengths[f] >= length(generation)
            frontlengths = frontlengths[1:f]
            break
        end
    end

    # Sort the last front by its crowding distance
    if frontlengths[end] > length(generation)
        fronts = sort_crowding_distance(generation, generationparent, fronts, frontlengths)
    end

    # Put solutions into the offspring generation
    for i in 1:length(generation)
        offspring[i] = population_idx(generation, generationparent, fronts[i])
    end

    return algo.currentminima, length(algo.seensolutions)
end

"""
Perform the non-dominated sort and return as many fronts as necessary to completey fill the offspring
generation. Calculating all the other fronts is unnecessary because they are discarded anyway.

:param algo: The algorithm.
:param generation: The offspring generation.
:param generationparent: The parent generation.
"""
function sort_nondominated(algo::Algorithm, generation::Generation, generationparent::Generation)
    # If current minimima have never been written, just set them to the reward of the first
    # solution to initialize them.
    if length(algo.currentminima) == 0
        algo.currentminima = generationparent[1].reward
    end

    reset_sorting_values!(generationparent)

    # Index directly into the two generation objects instead of concatenating them
    length_generation = length(generation)
    length_population::Int = length_generation * 2
    # Determine comparison functions for reward evaluation.
    smaller(sol1, sol2) = all(sol1.reward .<= sol2.reward) && any(sol1.reward .< sol2.reward)
    greater(sol1, sol2) = all(sol1.reward .>= sol2.reward) && any(sol1.reward .> sol2.reward)

    checkmin(reward, idx) = reward[idx] < algo.currentminima[idx]
    checkmax(reward, idx) = reward[idx] > algo.currentminima[idx]

    # Determine which functions to use for comparisons, depending on the optimization sense
    # of the algorith (minimize vs. maximize).
    if algo.sense == "minimize"
        comp_first = smaller
        comp_second = greater
        check = checkmin
    else
        comp_first = greater
        comp_second = smaller
        check = checkmax
    end

    fronts = Array{Int}(undef, length_population)
    frontlengths = Int[0]
    for i in 1:length_population
        # Compare solutions and assign domination values
        sol1 = population_idx(generation, generationparent, i)
        for j in 1:length_population
            if i == j
                continue
            end
            sol2 = population_idx(generation, generationparent, j)
            if comp_first(sol1, sol2)
                push!(sol1.dominates, j)
            elseif comp_second(sol1, sol2)
                sol1.dominatedby += 1
            end
        end

        # Figure out the current minimum/maximum values and store them.
        if sol1.dominatedby == 0
            for r in eachindex(sol1.reward)
                if check(sol1.reward, r)
                    algo.currentminima[r] = sol1.reward[r]
                end
            end

            # If the solution is not dominated by any other solution, push it into the
            # first front right away.
            frontlengths[1] += 1
            fronts[frontlengths[1]] = i
            sol1.rank = 1
        end
    end

    # Sort solutions into fronts
    leftsolutions = length_population - frontlengths[1]
    while leftsolutions > 0
        if (length(frontlengths) == 1 && frontlengths[end] != 0) || frontlengths[end] != frontlengths[end-1]
            push!(frontlengths, frontlengths[end])
        end

        f = length(frontlengths)
        beginning = f > 2 ? frontlengths[end-2] + 1 : 1
        for idx1 in fronts[beginning:frontlengths[end-1]]
            sol1 = population_idx(generation, generationparent, idx1)
            for idx2 in sol1.dominates
                sol2 = population_idx(generation, generationparent, idx2)
                sol2.dominatedby -= 1
                if sol2.dominatedby == 0
                    sol2.rank = f
                    frontlengths[end] += 1
                    fronts[frontlengths[end]] = idx2
                end
            end
        end

        leftsolutions = length_population - frontlengths[end]
    end

    return fronts, frontlengths
end

"""
Sort a front by crowding distance. Take the front and sort it by crowding distance
for all optimisation objectives. Return as many solutions as requested by length.

:param generation: Main generation.
:param generationparent: Parent generation.
:param fronts: List of solution indices sorted by fronts.
:param frontlengths: Lengths of each front (start and end points inside the fronts list).
"""
function sort_crowding_distance(generation::Generation, generationparent::Generation, fronts, frontlengths)
    # Make sure that there are enough solutions to sort through
    if length(generation) <= 2
        return fronts
    end

    getsolution(idx) = population_idx(generation, generationparent, idx)
    getreward(solutionidx, rewardidx) = getsolution(solutionidx).reward[rewardidx]

    # Perform the sort and crowding distance assignment for each objective
    beginning = length(frontlengths) > 1 ? frontlengths[end-1] : 1
    for i in eachindex(generation[1].reward)
        sorted = sort(fronts[beginning:frontlengths[end]], by=x -> getreward(x, i))

        maxmindist = getreward(sorted[end], i) - getreward(sorted[1], i)
        if maxmindist <= 0
            @warn "Crowding distance sort failed because all solutions are equal in dimension $i of reward."
            return fronts
        end

        getsolution(sorted[1]).crowdingdistance = Inf64
        getsolution(sorted[end]).crowdingdistance = Inf64

        for idx in 2:length(sorted)-1
            getsolution(sorted[idx]).crowdingdistance +=
                (getreward(sorted[idx-1], i) + getreward(sorted[idx+1], i)) / maxmindist
        end
    end

    # Sort solutions by crowdingdistance
    sort!(fronts[frontlengths[end-1]:frontlengths[end]], by=x -> getsolution(x).crowdingdistance, rev=true)
    return fronts
end
end
