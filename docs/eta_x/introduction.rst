.. _intro_etax:

Introduction
=============
*eta_x* is the rolling horizon optimization module which combines the functionality of the
other modules. It is based on the OpenAI *gym* framework and utilizes algorithms and functions
from the `stable_baselines3 <https://stable-baselines3.readthedocs.io/>`_ package. The *eta_x* module
also contains some extensions for *stable_baselines3*, these include additional policies, extractors,
schedules and agents.

The module contains functions meant to simplify the general process of creating rolling horizon
optimization models. It contains the *ETAx* class which in turn combines all of this information
such that you can start simple optimizations in just two lines. For example, to start the pendulum
example (which is taken from the *gym* framework):

.. literalinclude:: /../examples/pendulum/main.py
    :start-after: --main--
    :end-before: --main--
    :dedent:

The resulting optimization will have full configuration support, logging, support for multiple
series of optimization runs and many other things.

The algorithms in *eta_x* are an extension to the algorithms provided by *stable_baselines3*.
These algorithms specifically include some algorithms which are not from the field of reinforcement
learning but can be employed in generalized rolling horizon optimization settings.

Some base classes for environments, available in eta_utility.eta_x.envs make it
easy to create new environments in many cases (for example when using FMU simulation models) or when
performing model predictive control using mathematical models. These environments take away the work
of setting up models, passing around step results and so on.

*eta_x* extends the general Markov Decision Process by the option to introduce interactions between
multiple environments. This enables the creation of digital twins, which could for example use
a mathematical or a simulation model for some aspects and interact with the actual devices for other
aspects.

Take a look at the examples folder in the *eta_utility* repository to see some of the possibilities.

What are series and runs?
---------------------------
*eta_x* build on the concept of experiments. An experiment can be configured to perform
optimizations using specific environments and agents. An example of this concept is shown in the figure

.. figure:: figures/Concept_state_config.png
    :scale: 15
    :alt: Example of the eta_x experiment concept

    Example of the eta_x experiment concept.

As shown in the figure, the process starts with the configuration (setup) file, which is written in
JSON format (see :ref:`eta_experiment_config`). Based on this configuration, the environment and
corresponding agent can be initialized and executed.

An experiment with a single configuration can consist of a series of different optimization runs.
Each optimization run could for example have different external conditions for the environment
(such as being performed at a different time of the year).

What is an algorithm or agent?
-----------------------------------
The control algorithm receives inputs from the environment and follows a strategy to control
a system. This strategy could be either rule based, determined by mathematical optimization,
machine learning (reinforcement learning) or other methods, such as metaheuristics.

The agent receives observations from an environment and determines actions to control the environment
based on those observations.

What is an environment?
--------------------------
The environment is a dynamic system, which receives inputs (actions) from the control algorithm.
Observations made in the environment are passed to the agent.

How to get started
--------------------
Usually you want to use the *ETAx* class as shown above to initialize your experiment.
This will automatically load a JSON configuration file (see also :ref: `etax_experiment_config`).
The file to load the configuration from is specified during class instantiation:

.. autoclass:: eta_utility.eta_x::ETAx

After the class is instantiated, you can use the play and learn methods to execute the experiment:

.. autofunction:: eta_utility.eta_x::ETAx.learn

.. autofunction:: eta_utility.eta_x::ETAx.play

.. _eta_experiment_config:

Experiment configuration
-------------------------
The central part of the eta_x module is the experiment configuration. This configuration can be
read  from a JSON file and determines the setup of the entire experiment, including which agent and
environment to load and how to set each one up. The configuration is defined by the *ConfigOpt*
dataclass and its subsidiaries *ConfigOptSetup* and *ConfigOptSettings*.

When you are using ETAx (the class) the configuration will be read automatically.

Use :func:`eta_utility.eta_x::ConfigOpt.from_json` to read the configuration from a JSON file:

.. autofunction:: eta_utility.eta_x::ConfigOpt.from_json

Config section 'setup'
^^^^^^^^^^^^^^^^^^^^^^^^
The settings configured in the setup section are the following:

.. autoclass:: eta_utility.eta_x::ConfigOptSetup
    :members:
    :noindex:
    :exclude-members: from_dict

Config section 'paths'
^^^^^^^^^^^^^^^^^^^^^^^^
The paths section can contain the following relative paths:

.. autoattribute:: eta_utility.eta_x::ConfigOpt.relpath_results
.. autoattribute:: eta_utility.eta_x::ConfigOpt.relpath_scenarios

Config section 'settings'
^^^^^^^^^^^^^^^^^^^^^^^^^^
The configuration options in the settings section are the following.

 .. note::
    The configuration options "environment_specific", "interaction_env_specific" and "agent_specific"
    are separate sections on the top level. They are loaded into the settings object as dictionaries.
    To determine, which options are valid for these sections, please look at the arguments required
    for instantiation of the agent or environment. These arguments must be specified as parameters in
    the corresponding section.

.. autoclass:: eta_utility.eta_x::ConfigOptSettings
    :members:
    :noindex:
    :exclude-members: from_dict

Configuration example
^^^^^^^^^^^^^^^^^^^^^^^^
The following is the configuration for the pendulum example in this repository. It is relatively
minimal in that it makes extensive use of the defaults defined in the *ConfigOpt* classes.

.. literalinclude:: /../examples/pendulum/pendulum_learning.json
    :language: json

Configuration for optimization runs
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
An optimization run must also be configured. This is done through the *ConfigOptRun* class. The
class uses the series name and run names for initialization. It provides facilities to
create the paths required for optimization and to store information about the environments.
Below, you can see the parameters that ConfigOptRum offers. Full documentation is in the API
docs: :py:class:`eta_utility.eta_x.config.ConfigOptRun`.

.. autoclass:: eta_utility.eta_x::ConfigOptRun
    :members:
    :noindex:
    :exclude-members: from_dict, set_env_info, set_interaction_env_info, create_results_folders
