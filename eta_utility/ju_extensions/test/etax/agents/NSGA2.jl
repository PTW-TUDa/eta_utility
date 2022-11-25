using Test
using PyCall

import Random: AbstractRNG, TaskLocalRNG, rand
import ju_extensions.Agents.Nsga2

@pyimport eta_utility.eta_x.agents.nsga2 as py_nsga2
@pyimport numpy as np

const RAND_REPEAT = 100

"""
Create a list of variables parameter objects with the specified type and variable info.

:param type: type of the variables to be created (int or float)
:param min: Minimum value of the variables
:param max: Maximum value of the variables
:param length: length of the list
"""
create_varparams(type::String, min::Int, max::Int, length::Int) =
    [py_nsga2._VariableParameters(type, min, max) for _ in 1:length]

"""
Check if all events are present in the events array (should be consecurive numbers from 0 to length).

:param length: Expected length of the array.
:param events: Events array.
"""
function allevents_present(len::Int, events)
    presentnums = falses(len)
    for i in events
        presentnums[i+1] = true
    end
    all(presentnums)
end

@testset "Solution" begin
    lenevents = 5
    lenvars = 1
    solution = Nsga2.Solution(lenevents, create_varparams("int", 0, 3, lenvars), Inf, false)

    # Test whether all class attributes are initialized correctly
    @test all(solution.events[i] == i - 1 for i in 1:lenevents)
    @test solution.reward == Float64[Inf]
    @test solution.dominatedby == 0
    @test solution.rank == typemax(Int)
    @test solution.crowdingdistance == 0

    # Test the getter functions for variables and events
    @test length(solution.variables) == lenvars
    @test length(solution.events) == lenevents
    @test length(Nsga2.Actions(solution)) == lenvars + lenevents
    @test length(collect(Nsga2.Actions(solution))) == lenvars + lenevents
end

@testset "Randvar" begin
    varparam = create_varparams("int", 0, 3, 1)[1]
    testvars = [Nsga2.randvar(TaskLocalRNG(), varparam) for i in 1:RAND_REPEAT]
    # Make sure they are not all the same
    @test !all([testvars[1] == v for v in testvars])
    # Check they are all within the bounds
    @test all(varparam.minimum <= v <= varparam.maximum for v in testvars)
end

@testset "RandomizeSolution, $i" for i in 1:RAND_REPEAT
    lenevents = 5
    lenvars = 5
    varparams = create_varparams("int", 0, 3, lenvars)
    solution = Nsga2.Solution(lenevents, varparams, Inf, false)

    previoushash = Nsga2.hash(solution)
    previousevents = copy(solution.events)
    previousvars = copy(solution.variables)

    Nsga2.randomize!(TaskLocalRNG(), solution, varparams)
    # Make sure the events are not all in order anymore
    @test !all(solution.events[i] == i - 1 for i in 1:lenevents)
    # Check whether all numbers are still in the events
    @test begin
        presentnums = falses(lenevents)
        for i in solution.events
            presentnums[i+1] = true
        end
        all(presentnums)
    end
    # Values should be different after randomization
    newhash = Nsga2.hash(solution)
    @test newhash != previoushash
    newevents = solution.events
    @test previousevents != newevents
    newvars = solution.variables
    @test previousvars != newvars
    # Values should remain the same, if the solution is not changed.
    @test newhash == Nsga2.hash(solution)
    @test newevents == solution.events
    @test newvars == solution.variables
end

struct MockRNG04 <: AbstractRNG end
rand(rng::MockRNG04) = 0.4

struct MockRNG09 <: AbstractRNG end
rand(rng::MockRNG09) = 0.9

@testset "MutationRates" begin
    rateevents, ratevars = Nsga2.distribute_rates(MockRNG04(), 20, 20, 0.2)
    @test rateevents ≈ 0.08
    @test ratevars ≈ 0.24

    rateevents, ratevars = Nsga2.distribute_rates(MockRNG09(), 20, 20, 0.2)
    @test rateevents ≈ 0.175
    @test ratevars ≈ 0.05
end

@testset "Mutation" begin
    mutationrate = 0.2
    lenevents = 20
    lenvars = 20

    varparams = create_varparams("int", 0, 3, lenvars)
    solution_parent = Nsga2.Solution(lenevents, varparams, Inf, false)
    solution = Nsga2.Solution(lenevents, varparams, Inf, false)

    Nsga2.randomize!(TaskLocalRNG(), solution_parent, varparams)
    Nsga2.mutate!(solution, solution_parent, TaskLocalRNG(), mutationrate, varparams)

    previousevents = solution_parent.events
    previousvars = solution_parent.variables

    # Make sure, the events are not the same after the mutation
    newevents = solution.events
    @test allevents_present(lenevents, newevents)
    @test previousevents != newevents
    differentevents = 0
    for i in eachindex(newevents)
        if newevents[i] != previousevents[i]
            differentevents += 1
        end
    end
    @test 2 <= differentevents <= mutationrate * lenevents * 2

    # Make sure, the variables are not the same after mutation
    newvars = solution.variables
    @test length(newvars) == lenvars
    @test previousvars != newvars
    differentvars = 0
    for i in eachindex(newvars)
        if newvars[i] != previousvars[i]
            differentvars += 1
        end
    end
    @test 1 <= differentvars <= mutationrate * lenvars * 2
    normally_changed = mutationrate * (lenvars + lenevents)
    @test normally_changed - 2 <= differentvars + differentevents <= normally_changed + 1
end

@testset "Crossover" begin
    crossoverrate = 0.2
    maxcrosslen = 10
    lenevents = 20
    lenvars = 20

    varparams = create_varparams("int", 0, 3, lenvars)
    solution_parent = Nsga2.Solution(lenevents, varparams, Inf, false)
    Nsga2.randomize!(TaskLocalRNG(), solution_parent, varparams)
    solution = Nsga2.Solution(lenevents, varparams, Inf, false)
    Nsga2.randomize!(TaskLocalRNG(), solution, varparams)

    parentevents = solution_parent.events
    parentvars = solution_parent.variables
    previousevents = copy(solution.events)
    previousvars = copy(solution.variables)

    Nsga2.crossover!(solution, solution_parent, TaskLocalRNG(), crossoverrate, maxcrosslen)

    #Make sure, the events are not the same after the mutation
    newevents = solution.events
    @test allevents_present(lenevents, newevents)
    @test previousevents != newevents != parentevents

    # Make sure, the variables are not the same after mutation
    newvars = solution.variables
    @test length(newvars) == lenvars
    @test previousvars != newvars != parentvars
end

@testset "Sorting" begin
    lenevents = 20
    lenvars = 20
    population = 10

    algo = Nsga2.Algorithm(
        population,
        0.2,
        0.2,
        10,
        10000,
        lenevents,
        create_varparams("int", 0, 3, lenvars),
        Inf,
        "minimize",
        1502,
    )

    varparams = create_varparams("int", 0, 3, lenvars)
    generation1 = Nsga2.create_generation(algo, false)
    generation2 = Nsga2.create_generation(algo, false)

    for idx in eachindex(generation1)
        Nsga2.randomize!(TaskLocalRNG(), generation1[idx], varparams)
        Nsga2.randomize!(TaskLocalRNG(), generation2[idx], varparams)

        generation1[idx].reward = [idx, length(generation1) - idx]
        generation2[idx].reward = [length(generation1) - idx + 1, idx + 1]
    end
    fronts, frontlengths = Nsga2.sort_nondominated(algo, generation1, generation2)
    @test length(fronts) == population * 2
    @test begin
        presentnums = falses(population * 2)
        for i in fronts
            presentnums[i] = true
        end
        all(presentnums)
    end
    @test length(frontlengths) <= population * 2
    @test maximum(frontlengths) == population * 2

    fronts = Nsga2.sort_crowding_distance(generation1, generation2, fronts, frontlengths)
    @test length(fronts) == population * 2
    @test begin
        presentnums = falses(population * 2)
        for i in fronts
            presentnums[i] = true
        end
        all(presentnums)
    end
end

@testset "SortingSingleReward" begin
    lenevents = 20
    lenvars = 20
    population = 10

    algo = Nsga2.Algorithm(
        population,
        0.2,
        0.2,
        10,
        10000,
        lenevents,
        create_varparams("int", 0, 3, lenvars),
        Inf,
        "minimize",
        1502,
    )

    varparams = create_varparams("int", 0, 3, lenvars)
    generation1 = Nsga2.create_generation(algo, false)
    generation2 = Nsga2.create_generation(algo, false)

    for idx in eachindex(generation1)
        Nsga2.randomize!(TaskLocalRNG(), generation1[idx], varparams)
        Nsga2.randomize!(TaskLocalRNG(), generation2[idx], varparams)

        generation1[idx].reward = [idx]
        generation2[idx].reward = [idx + length(generation1)]
    end
    fronts, frontlengths = Nsga2.sort_nondominated(algo, generation1, generation2)
    @test length(fronts) == population * 2
    @test begin
        presentnums = falses(population * 2)
        for i in fronts
            presentnums[i] = true
        end
        all(presentnums)
    end
    @test length(frontlengths) <= population * 2
    @test maximum(frontlengths) == population * 2

    fronts = Nsga2.sort_crowding_distance(generation1, generation2, fronts, frontlengths)
    @test length(fronts) == population * 2
    @test begin
        presentnums = falses(population * 2)
        for i in fronts
            presentnums[i] = true
        end
        all(presentnums)
    end
end
