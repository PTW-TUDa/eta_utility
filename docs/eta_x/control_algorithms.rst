.. _agents:

Control Algorithms
=======================
The agents implemented in *eta_x.agents* are subclasses of
:py:class:`stable_baselines3.common.base_class.BaseAlgorithm` in *stable_baselines3*. Calling them agents
is a remnant from *stable_baselines2* - the wording was changed in *stable_baselines3*).

Usually there is no need to dive more deeply into the agents provided by *eta_x*. You can use them by specifying
their import path in your experiment configuration and don't have to worry about how they work. It is good to know
however, that some agents do not implement all methods which would be required by the interface in normal usage
within the *eta_x* framework this should not be a problem.

The currently available agents are listed here. Note that you need to specify the parameters required for
instantiation in the *agent_specific* section of the *eta_x* configuration file.

Model Predictive Control (MPC) Agent
---------------------------------------
The MPC agent implements a model predictive controller. It can be used to execute mathematical models in conjunction
with mathematical solvers such as cplex or glpk and it relies on the *pyomo* library to achieve this.

You can provide additional arguments in *kwargs* to the agent. These will be interpreted first as arguments for
the base class and then for the solver. Meaning that arguments which are passed to MPCBasic and not recognized
by *BaseAlgorithm* will be passed on to the solver. This allows free configuration of all solver options.

.. autoclass:: eta_utility.eta_x.agents::MPCBasic
    :noindex:

Rule Based Agent (Base Class)
---------------------------------
The rule based agent is a base class which facilitates the creation of simple rule based agents. To use it you need
to implement the :py:class:`eta_utility.eta_x.agents.RuleBased.control_rules` method. The control_rules method
takes the array of observations from the environment and determines an array of actions based on them.

.. autoclass:: eta_utility.eta_x.agents::RuleBased
    :members: control_rules
    :noindex: