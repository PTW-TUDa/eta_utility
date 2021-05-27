""" The FMUSimulator class enables easy simulation of FMU files.
"""
import pathlib
import shutil
from datetime import timedelta
from typing import Dict, Mapping, Sequence

import numpy as np
from fmpy import extract, read_model_description
from fmpy.fmi2 import FMU2Model, FMU2Slave
from fmpy.simulation import apply_start_values
from fmpy.sundials import CVodeSolver

from eta_utility import get_logger

log = get_logger("simulators.FMUSimulator")


class FMUSimulator:
    """FMU simulator object

    :param int _id: FMU instance id
    :param fmu_path: Path to the FMU file
    :type fmu_path: str or pathlib.Path
    :param start_time: Simulation start time in seconds
    :type start_time: float or timedelta
    :param float stop_time: Simulation stop time in seconds
    :type stop_time: float or timedelta
    :param step_size: simulation step size in seconds
    :type step_size: float or timedelta
    :param names_inputs: List of input names that correspond to names used in the FMU file (e.g. ['u', 'p'])
    :type names_inputs: Sequence[str]
    :param names_outputs: List of output names that correspond with names used in
                          the FMU file (e.g. ['y', 'th', 'thdot'])
    :type names_outputs: Sequence[str]
    :param init_values: Starting values for parameters that should be pushed to the FMU with names corresponding to
                        variables in the FMU
    :type init_values:  Mapping[str, float]
    """

    def __init__(self, _id, fmu_path, start_time, stop_time, step_size, names_inputs, names_outputs, init_values=None):
        # read the model description and input/outputs
        self.path_to_FMU = fmu_path
        self.start_time = start_time
        self.stop_time = stop_time  # not needed though in the following
        self.step_size = step_size
        self.names_inputs = names_inputs
        self.names_outputs = names_outputs

        # read model description from FMU (contains variable names, types, references)
        self.model_description = read_model_description(fmu_path)

        # iterate through all FMU variables and save indices of names_inputs, names_outputs
        self.variable_references = {}
        self.variable_types = {}
        for variable in self.model_description.modelVariables:
            self.variable_references[variable.name] = variable.valueReference
            self.variable_types[variable.name] = variable.type

        # iterate through all names_inputs and save corresponding mapping and FMU valueReference
        self.typemapping_inputs = {"Real": [], "Integer": [], "Boolean": []}
        self.references_inputs = {"Real": [], "Integer": [], "Boolean": []}
        for idx, name in enumerate(names_inputs):
            variable_reference = self.variable_references[name]
            variable_type = self.variable_types[name]
            if variable_reference is not None and variable_type is not None:
                self.typemapping_inputs[variable_type].append(idx)
                self.references_inputs[variable_type].append(variable_reference)
            else:
                log.warning(
                    "Input variable '{}' couldn't be found in FMU model description. "
                    "Entry will be ignored.".format(name)
                )

        # iterate through all names_outputs and save corresponding mapping and FMU valueReference
        self.typemapping_outputs = {"Real": [], "Integer": [], "Boolean": []}
        self.references_outputs = {"Real": [], "Integer": [], "Boolean": []}
        for idx, name in enumerate(names_outputs):
            variable_reference = self.variable_references[name]
            variable_type = self.variable_types[name]
            if variable_reference is not None and variable_type is not None:
                self.typemapping_outputs[variable_type].append(idx)
                self.references_outputs[variable_type].append(variable_reference)
            else:
                log.warning(
                    "Output variable '{}' couldn't be found in FMU model description."
                    "Entry will be ignored.".format(name)
                )

        # count number of inputs and outputs
        self.n_inputs = {
            "Real": len(self.references_inputs["Real"]),
            "Integer": len(self.references_inputs["Integer"]),
            "Boolean": len(self.references_inputs["Boolean"]),
        }
        self.n_outputs = {
            "Real": len(self.references_outputs["Real"]),
            "Integer": len(self.references_outputs["Integer"]),
            "Boolean": len(self.references_outputs["Boolean"]),
        }
        self.n_outputs_sum = self.n_outputs["Real"] + self.n_outputs["Integer"] + self.n_outputs["Boolean"]

        # extract the FMU
        self.unzipdir = extract(fmu_path)
        self.fmu = FMU2Slave(
            guid=self.model_description.guid,
            unzipDirectory=self.unzipdir,
            modelIdentifier=self.model_description.coSimulation.modelIdentifier,
            instanceName="FMUsimulator_" + str(_id),
        )

        # initialize
        self.fmu.instantiate(visible=False, callbacks=None, loggingOn=False)
        self.fmu.setupExperiment(startTime=self.start_time)
        self.fmu.enterInitializationMode()

        init_values = {} if init_values is None else init_values
        apply_start_values(self.fmu, self.model_description, start_values=init_values, apply_default_start_values=False)

        self.fmu.exitInitializationMode()
        self.time = start_time

    def step(self, input_values, advance_time=True):
        """Simulate next time step in the FMU with defined input values

        :param input_values: Current values that should be pushed to the FMU with the names corresponding to
                             variables in the FMU
        :type input_values: Mapping[str, float]
        :param bool advance_time: To decide if the FMUsimulator should add one timestep to the simulation time or not
                                  This has to be done elsewhere, if this is deactivated
        :return: Resulting input and output values from the FMU with the keys named corresponding to the variables
                 in the FMU
        :rtype: Dict[str, float]
        """

        # convert input_values to numpy array
        input_values = np.array(input_values)

        # set input values of type 'Real'
        if self.n_inputs["Real"] > 0:
            input_values_real = list(input_values[self.typemapping_inputs["Real"]])
            self.fmu.setReal(self.references_inputs["Real"], input_values_real)

        # set input values of type 'Integer'
        if self.n_inputs["Integer"] > 0:
            input_values_integer = list(input_values[self.typemapping_inputs["Integer"]])
            self.fmu.setInteger(self.references_inputs["Integer"], input_values_integer)

        # set input values of type 'Boolean'
        if self.n_inputs["Boolean"] > 0:
            input_values_boolean = list(input_values[self.typemapping_inputs["Boolean"]])
            self.fmu.setBoolean(self.references_inputs["Boolean"], input_values_boolean)

        # push input values to the FMU and do one timestep, doStep performs a step of certain size
        self.fmu.doStep(currentCommunicationPoint=self.time, communicationStepSize=self.step_size)

        # create empty output_values numpy array
        output_values = np.zeros(self.n_outputs_sum)

        # get output values of type 'Real' and save into mapped output_values
        if self.n_outputs["Real"] > 0:
            output_values_real = self.fmu.getReal(self.references_outputs["Real"])
            output_values[self.typemapping_outputs["Real"]] = np.array(output_values_real)

        # get output values of type 'Integer' and save into mapped output_values
        if self.n_outputs["Integer"] > 0:
            output_values_integer = self.fmu.getReal(self.references_outputs["Integer"])
            output_values[self.typemapping_outputs["Integer"]] = np.array(output_values_integer)

        # get output values of type 'Boolean' and save into mapped output_values
        if self.n_outputs["Boolean"] > 0:
            output_values_boolean = self.fmu.getReal(self.references_outputs["Boolean"])
            output_values[self.typemapping_outputs["Boolean"]] = np.array(output_values_boolean)

        # advance the time
        if advance_time is True:
            self.time += int(self.step_size)

        return list(output_values)

    def reset(self, init_values):
        """Reset FMU to specified initial condition

        :param init_values: Values for initialization
        :type init_values: Mapping[str, float]
        """
        self.time = self.start_time
        self.fmu.reset()
        self.fmu.setupExperiment(startTime=self.start_time)
        self.fmu.enterInitializationMode()

        apply_start_values(
            self.fmu, self.model_description, start_values=init_values, apply_default_start_values=False
        )  # new

        self.fmu.exitInitializationMode()

    def close(self):
        """Close the FMU and tidy up the unzipped files"""
        self.fmu.terminate()
        self.fmu.freeInstance()
        shutil.rmtree(self.unzipdir)  # clean up unzipped files


class FMU2_ME_Slave(FMU2Model):
    """Helper class for simulation of FMU2 FMUs. This is as wrapper for FMU2Model.
    It can be used to wrap model exchange FMUs such that they can be simulated similar to a co-simulation FMU. THis
    is especially helpful for testing model exchange FMUs.

    It exposes an interface that emulates part of the original FMU2Slave class from fmpy.
    """

    # Define some constants that might be needed according to the FMI Standard
    fmi2True = 1
    fmi2False = 0

    fmi2OK = 0
    fmi2Warning = 1
    fmi2Discard = 2
    fmi2Error = 3
    fmi2Fatal = 4
    fmi2Pending = 5

    def __init__(self, **kwargs) -> None:
        """Initialize the FMU2Slave object

        .. seealso:: fmpy.fmi2.FMU2Model

        :param Any **kwargs: Accepts any parameters that fmpy.FMU2Model accepts.
        """
        super().__init__(**kwargs)
        self._model_description = read_model_description(kwargs["unzipDirectory"])
        self._solver = None
        self._tolerance = 0.0
        self._stop_time = 0.0
        self._start_time = 0.0

    def setupExperiment(self, tolerance: float = None, startTime: float = 0.0, stopTime: float = None, **kwargs) -> int:
        """Experiment setup and storage of required values.

        .. seealso:: fmpy.fmi2.FMU2Model.setupExperiment

        :param float tolerance: Solver tolerance, default value is 1e-5
        :param float startTime: Starting time for the experiment
        :param float stopTime: Ending time for the experiment
        :param Any kwargs: Other keyword arguments that might be required for FMU2Model.setupExperiment in the future.
        :return: FMI2 return value
        :rtype: int
        """
        self._tolerance = 1e-5 if tolerance is None else tolerance
        self._stop_time = 0.0 if stopTime is None else stopTime
        self._start_time = startTime
        self._stop_time = stopTime

        kwargs["tolerance"] = tolerance
        kwargs["stopTime"] = stopTime
        kwargs["startTime"] = startTime

        return super().setupExperiment(**kwargs)

    def exitInitializationMode(self, **kwargs) -> int:
        """Exit the initialization mode and setup the cvode solver.

        .. seealso:: fmpy.fmi2.FMU2Model.exitInitializationMode

        :param Any kwargs: Keyword arguments accepted by FMU2Model.exitInitializationMode
        :return: FMI2 return value
        :rtype: int
        """
        ret = super().exitInitializationMode(**kwargs)

        # Collect discrete states from FMU
        self.eventInfo.newDiscreteStatesNeeded = self.fmi2True
        self.eventInfo.terminateSimulation = self.fmi2False

        while (
            self.eventInfo.newDiscreteStatesNeeded == self.fmi2True
            and self.eventInfo.terminateSimulation == self.fmi2False
        ):
            # update discrete states
            self.newDiscreteStates()
        self.enterContinuousTimeMode()

        # Initialize solver
        self._solver = CVodeSolver(
            set_time=self.setTime,
            startTime=self._start_time,
            maxStep=(self._stop_time - self._start_time) / 50.0,
            relativeTolerance=self._tolerance,
            nx=self._model_description.numberOfContinuousStates,
            nz=self._model_description.numberOfEventIndicators,
            get_x=self.getContinuousStates,
            set_x=self.setContinuousStates,
            get_dx=self.getDerivatives,
            get_z=self.getEventIndicators,
        )

        return ret

    def doStep(
        self,
        currentCommunicationPoint: float,
        communicationStepSize: float,
        noSetFMUStatePriorToCurrentPoint: int = None,
    ) -> int:
        """Perform a simulation step. Advance simulation from currentCommunicationPoint by communicationStepSize

        .. seealso: FMI2 Standard documentation

        :param float currentCommunicationPoint: current time stamp (starting point for simulation step)
        :param float communicationStepSize: time step size
        :param int noSetFMUStatePriorToCurrentPoint: Determine whether a reset before the currentCommunicationPoint is
                                                     possible. Must be either fmi2True or fmi2False
        :return: FMU2 return value
        :rtype: int
        """
        time = currentCommunicationPoint
        step_size = communicationStepSize
        set_fmu_state = (
            noSetFMUStatePriorToCurrentPoint if noSetFMUStatePriorToCurrentPoint is not None else self.fmi2True
        )

        # Perform a solver step and reset the FMU Model time.
        _, time = self._solver.step(time, time + step_size)
        self.setTime(time)
        # Check for events that might have occured during the step
        step_event, _ = self.completedIntegratorStep()

        return self.fmi2OK
