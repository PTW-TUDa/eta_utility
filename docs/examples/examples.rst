.. _examples:

Usage examples
================
*eta_utility* contains example implementations for different usages of the package.
This page gives a short overview of the examples.

Connectors
--------------
There are two examples for the *connectors*: The *read_series_eneffco* example illustrates a simple usage of the *connectors* module. It selects some data points
and reads them as series data from the server.

The *data_recorder* example is more complex in that it uses multiple *connectors*,
can connect to different protocols and provides a command line interface for
configuration.

eta_x Optimization
--------------------
There also examples for the optimization part of the framework. The *pendulum* example is the simplest one of them. It implements an inverse pendulum, similar to
the `equivalent example in OpenAI gym <https://gym.openai.com/envs/Pendulum-v0/>`_.
The environment can be used for
different kinds of agents and includes examples for the PPO reinforcement learning
agent as well as a simple rule based controller.

The *damped_oscillator* example illustrates how simulation environments are created,
based on the *BaseEnvSim* class. In this simple example only the StateConfig and the
render function needs to be specified to obtain a completely functional environment.
In the example the controller will just supply random action values.

Finally, the *cyber_physical_system* example shows the full capabilities of the *eta_utility*
framwork. It utilizes the interaction between a simulation and an actual machine to
supply advanced observations to an agent which controls the tank heating unit of
an industrial parts cleaning machine.