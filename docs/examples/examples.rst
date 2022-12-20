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

Forecasting
--------------
The *forecasting* example demonstrates how a machine learning model can be deployed using
the *connectors* module to connect to data sources (in this example OPC UA). The model is a load forecasting
model of a machine tool in the ETA research factory. It forecasts the electric load of the machine tool during
the next 100 seconds. Input data is a time window of the last 100 seconds of 8 separate electric load signals
of sub-components of the machine tool as well as the total electric load of the machine tool. The model was
exported into ONNX format and is deployed using the *onnxruntime* module. The forecast is published to an
OPC UA server that is established on the local machine using the *servers* module.

In the example, data is collected for 100 seconds in an internal memory. Once enough time steps are present, the
inference is triggered and the forecast is published to the OPC UA server. This loop is repeated every second.

eta_x Optimization
--------------------
Examples for the optimization part of the framework are also provided. The *pendulum* example is the
simplest one of them. It implements an inverse pendulum, similar to
the `equivalent example in OpenAI gym <https://gym.openai.com/envs/Pendulum-v0/>`_.
The environment can be used for
different kinds of agents and includes examples for the PPO reinforcement learning
agent as well as a simple rule based controller.

The *damped_oscillator* example illustrates how simulation environments are created,
based on the *BaseEnvSim* class. In this simple example, only the StateConfig and the
render function need to be specified to obtain a completely functional environment.
In the example, the controller will just supply random action values.

Finally, the *cyber_physical_system* example shows the full capabilities of the *eta_utility*
framework. It utilizes the interaction between a simulation and an actual machine to
supply advanced observations to an agent which controls the tank heating unit of
an industrial parts cleaning machine. To be able to run this example, a Dymola license is needed.
In the :ref:`dymola_license_not_found` it is explained how to use the license.
