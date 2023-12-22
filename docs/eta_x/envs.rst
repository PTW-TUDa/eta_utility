.. _envs:

Environments
===============
*eta_utility* environments are based on the interfaces offered by `stable_baselines3
<https://stable-baselines3.readthedocs.io/en/master/guide/custom_env.html>`_ which are in turn based on the
`Farama gymnasium environments <https://gymnasium.farama.org/api/env/>`_. The *eta_x* environments are provided as
abstract classes which must be subclassed to create useful implementations. For the specific use cases they are
intended for, these base classes make the creation of new environments much easier.

Custom environments should follow the interface for custom environments discussed in the `stable_baselines3
documentation <https://stable-baselines3.readthedocs.io/en/master/guide/custom_env.html>`_. The following describes
the functions available to simplify implementation of specific functionality in custom environments. You can look
at the :ref:`examples` for some inspiration what custom environments can look like.

The custom environments created with the utilities described here can be used directly with *stable_baselines3* or
*gymnasium*. However, using the :py:class:`eta_utility.eta_x::ETAx` class is recommended (see :ref:`intro_etax`).
When using the *ETAx* class for your optimization runs, the parameters required for environment instantiation must
be configured in the *environment_specific* section of the configuration. If interaction between environments is also
configured, additional parameters can be set in the configuration file. To configure the interaction environment, use
the section *interaction_env_specific*. If that section is not present, the parameters from the *environment_specific*
section will be used for both environments.

Environment State Configuration
--------------------------------

The most important concept to understand when working with the environment utilities provided by *eta_utility* is
is the handling and configuration of the environment state. The state is represented by
:py:class:`eta_utility.eta_x.envs::StateVar` objects which each correspond to one variable of the environment. All
StateVar objects of an environment are combined into the StateConfig object. From the StateConfig object we can
determine most other aspects of the environment, such as for example the observation space and action space. The
*gymnasium* documentation provides more information about `Spaces <https://gymnasium.farama.org/api/spaces/>`_.

Each state variable is represented by a *StateVar* object:

.. autoclass:: eta_utility.eta_x.envs::StateVar
    :members:
    :noindex:
    :exclude-members: from_dict

    For example, the variable "tank_temperature" might be part of the environment's state. Let's assume it
    represents the temperature inside the tank of a cleaning machine. This variable could be read from an
    external source. In this case it must have ``is_ext_output = True`` and the name of the external variable
    to read from must be specified: ``ext_id = "T_Tank"``. If this value should also be passed to the agent as an
    observation, set ``is_agent_observation = True``. For observations and actions, you also need to set the
    low and high values, which determine the size of the observation and action spaces in this case something like
    ``low_value = 20`` and ``high_value = 80`` (if we are talking about water temperature measured in Celsius)
    might make sense.

    If you want the environment to safely abort the optimization when certain values are exceeded, set the abort
    conditions to sensible values such as ``abort_condition_min = 0`` and ``abort_condition_max = 100``. This
    can be especially useful for example if you have simulation models which do not support certain values
    (for example, in this case they might not be able to handle water temperatures higher than 100 °C)::

        v1 = StateVar(
            "tank_temperature",
            ext_id = "T_Tank",
            is_ext_output = True,
            is_agent_observation = True,
            low_value = 20,
            high_value = 80,
            abort_condition_min = 0,
            abort_condition_max = 100,
        )


    As another example, you could set up an agent action named ``name = "set_heater"`` which the environment uses
    to set the state of the tank heater. In this case, the state variable should be configured
    with ``is_agent_action = True`` and you might want to pass this on to a simulation model or an actual machine by
    setting ``is_ext_input = True``::

        v2 = StateVar(
            "set_heater",
            ext_id = "u_tank",
            is_ext_input = True,
            is_agent_action = True,
        )

    Finally, let's create a third variable which is read from a scenario file and converted from kilowatts to watts
    (multiplied by 1000). Additionally, this variable needs to be offset by a value of -10 due to measurement errors::

        v3 = StateVar(
            "outside_temperature",
            scenario_id = "T_ouside",
            scenario_scale_add = -10,
            scenario_scale_mult = 1000,
            is_agent_observation = True,
            low_value = 0,
            high_value = 40,
        )

All state variables are combined into the *StateConfig* object:

.. autoclass:: eta_utility.eta_x.envs::StateConfig
    :members:
    :noindex:
    :exclude-members: loc, from_dict,

    Using the examples above, we could create the *StateConfig* object by passing our three state variables to
    the constructor::

        state_config = StateConfig(v1, v2, v3)

    If you are creating an environment, assign the *StateConfig* object to ``self.state_config``. This will sometimes
    even be sufficient to create a fully functional environment.

The state config object and its attributes (such as the observations) are used by the environments to determine
which values to update during steps, which values to read from scenario files and which values to pass to the agent
as actions.

Base Environment
------------------

.. autoclass:: eta_utility.eta_x.envs::BaseEnv
    :members:
    :private-members:
    :inherited-members: abc.ABC
    :show-inheritance:
    :exclude-members: reward_range, metadata, spec, _seed, _abc_impl
    :noindex:

Model Predictive Control (MPC) Environment
------------------------------------------------
The BaseEnvMPC is a class for the optimization of mathematical MPC models.

.. autoclass:: eta_utility.eta_x.envs::BaseEnvMPC
    :members:
    :private-members:
    :inherited-members: abc.ABC
    :show-inheritance:
    :exclude-members: reward_range, metadata, spec, _seed, _init_legacy, _init_state_space, _abc_impl
    :noindex:

Simulation (FMU) Environment
-----------------------------
The BaseEnvSim supports the optimization of FMU simulation models. Make sure to set the *fmu_name* attribute when
subclassing this environment. The FMU file will be loaded from the same directory as the environment itself.

.. autoclass:: eta_utility.eta_x.envs::BaseEnvSim
    :members:
    :private-members:
    :inherited-members: abc.ABC
    :show-inheritance:
    :exclude-members: reward_range, metadata, spec, _seed, _init_legacy, _init_state_space, _abc_impl
    :noindex:

Live Connection Environment
-----------------------------
The BaseEnvLive is an environment which create direct (live) connections to actual devices. It utilizes
:py:class:`eta_utility.connectors.LiveConnect` to achieve this. Please also read the corresponding documentation
because LiveConnect needs additional configuration.

.. autoclass:: eta_utility.eta_x.envs::BaseEnvLive
    :members:
    :private-members:
    :inherited-members: abc.ABC
    :show-inheritance:
    :exclude-members: reward_range, metadata, spec, _seed, _init_legacy, _init_state_space, _abc_impl
    :noindex:
