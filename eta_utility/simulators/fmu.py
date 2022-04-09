""" The FMUSimulator class enables easy simulation of FMU files.
"""
from __future__ import annotations

import itertools as it
import shutil
from datetime import timedelta
from typing import TYPE_CHECKING, Mapping

import numpy as np
from fmpy import extract, read_model_description
from fmpy.fmi2 import FMU2Model, FMU2Slave
from fmpy.simulation import apply_start_values
from fmpy.sundials import CVodeSolver
from fmpy.util import compile_platform_binary

from eta_utility import get_logger

if TYPE_CHECKING:
    from typing import Any, Sequence

    from fmpy.model_description import ModelDescription

    from eta_utility.type_hints import Number, Path, TimeStep

log = get_logger("simulators.fmu")


class FMUSimulator:
    """FMU simulator object.

    :param _id: FMU instance ID.
    :param fmu_path: Path to the FMU file.
    :param start_time: Simulation start time in seconds.
    :param float stop_time: Simulation stop time in seconds.
    :param step_size: Simulation step size in seconds.
    :param names_inputs: List of input names that correspond to names used in the FMU file (e.g. ['u', 'p']).
                         If the step function is going to be used with lists as input values, this list will be used
                         to translate between the list position and the variable name in the FMU.
    :param names_outputs: List of output names that correspond to names used in the FMU file
                          (e.g. ['y', 'th', 'thdot']). If the step function should return only specific values instead
                          of all results as a dictionary, this parameter can be specified to determine, which parameters
                          should be returned.
    :param init_values: Starting values for parameters that should be pushed to the FMU with names corresponding to
                        variables in the FMU.
    :param str return_type: "dict" or "list". Alter the standard behavior, which is to return lists from the step and
                            get_values functions only if both, "names_inputs" and "names_outputs" are specified.
                            This parameter will force the step and get_values functions to always return either
                            dictionaries or lists.
    """

    def __init__(
        self,
        _id: int,
        fmu_path: Path,
        start_time: TimeStep = 0,
        stop_time: TimeStep = 1,
        step_size: TimeStep = 1,
        names_inputs: Sequence[str] | None = None,
        names_outputs: Sequence[str] | None = None,
        init_values: Mapping[str, float] | None = None,
        *,
        return_type: str | None = None,
    ) -> None:
        #: Path to the FMU model.
        self.fmu_path = fmu_path

        #: Start time for the simulation in time increments.
        self.start_time = start_time.total_seconds() if isinstance(start_time, timedelta) else start_time
        #: Stopping time for the simulation in time increments (only relevant if run in simulation loop).
        self.stop_time = stop_time.total_seconds() if isinstance(stop_time, timedelta) else stop_time
        #: Step size (time) for the simulation in time increments.
        self.step_size = step_size.total_seconds() if isinstance(step_size, timedelta) else step_size

        #: Model description from the FMU (contains variable names, types, references and more).
        self.model_description: ModelDescription = read_model_description(fmu_path)

        #: Variable map from model description. The map specifies the value reference and datatype of a named
        #: variable in the FMU. The structure is {'name': {'ref': <value reference>, 'type': <variable data type>}}.
        self._model_vars: dict[str, dict[str, str]] = {}
        self.__type_map = {"Real": "real", "Boolean": "bool", "Integer": "int", "Enumeration": "enum"}

        for var in self.model_description.modelVariables:
            self._model_vars[var.name] = {"ref": var.valueReference, "type": self.__type_map[var.type]}

        #: Map of input variables which can be used to evaluate an ordered list of input variables. This is typically
        #: not required when working with mappings/dictionaries as step inputs.
        #:
        #: The map contains the following lists:
        #:
        #:     * real: Mask for real variables. This can be used to identify real variables from the complete set of
        #:       input variables (_inputs["refs"], see below) using `itertools.compress`.
        #:     * int: Mask for integer variables. This can be used to identify integer variables from the complete set
        #:       of input variables (_inputs["refs"], see below) using `itertools.compress`.
        #:     * bool: Mask for boolean variables. This can be used to identify boolean variables from the complete set
        #:       of input variables (_inputs["refs"], see below) using `itertools.compress`.
        self._input_map: dict[str, list[bool]] = {"names": [], "real": [], "int": [], "bool": []}

        #: Map of input variable references and their names. The map contains the following lists:
        #:
        #:     * names: List of the named input variables that are accessible in the model.
        #:     * refs: List of all value references to input variables of all types. This is the complete list, which
        #:       can be filtered using itertools.compress (see above).
        #:     * real: List of all value references to input variables of type real.
        #:     * int: List of all value references to input variables of type integer.
        #:     * bool: List of all value references to input variables of type boolean.
        self._inputs: dict[str, list[str]] = {"names": [], "refs": [], "real": [], "int": [], "bool": []}
        refs = []
        names = []
        iterator = names_inputs if names_inputs is not None else self._model_vars.keys()

        for idx, var in enumerate(iterator):
            if var in self._model_vars:
                refs.append(self._model_vars[var]["ref"])
                names.append(var)
                self._input_map["real"].append(self._model_vars[var]["type"] == "real")
                self._input_map["int"].append(self._model_vars[var]["type"] == "int")
                self._input_map["bool"].append(self._model_vars[var]["type"] == "bool")
            else:
                log.warning(
                    f"Input variable '{var}' couldn't be found in FMU model description. Entry will be ignored."
                )

        self._inputs["names"] = names
        self._inputs["refs"] = refs
        self._inputs["real"] = list(it.compress(refs, self._input_map["real"]))
        self._inputs["int"] = list(it.compress(refs, self._input_map["int"]))
        self._inputs["bool"] = list(it.compress(refs, self._input_map["bool"]))

        #: Map of output variables which can be used to evaluate an ordered list of output variables. This is typically
        #: not required when working with mappings/dictionaries as step outputs.
        #:
        #: The map contains the following lists:
        #:
        #:     * real: Mask for real variables. This can be used to identify real variables from the complete set of
        #:       output variables (_outputs['refs'], see below) using `itertools.compress`.
        #:     * int: Mask for integer variables. This can be used to identify integer variables from the complete set
        #:       of output variables (_outputs['refs'], see below) using `itertools.compress`.
        #:     * bool: Mask for boolean variables. This can be used to identify boolean variables from the complete set
        #:       of output variables (_outputs['refs'], see below) using `itertools.compress`.
        self._output_map: dict[str, list[bool]] = {"names": [], "real": [], "int": [], "bool": []}

        #: Map of output variable references and their names. The map contains the following lists:
        #:
        #:     * names: List of the named output variables that are accessible in the model.
        #:     * refs: List of all value references to output variables of all types. This is the complete list, which
        #:       can be filtered using itertools.compress (see above).
        #:     * real: List of all value references to output variables of type real.
        #:     * int: List of all value references to output variables of type integer.
        #:     * bool: List of all value references to output variables of type boolean.
        self._outputs: dict[str, list[str]] = {"names": [], "refs": [], "real": [], "int": [], "bool": []}
        refs = []
        names = []
        iterator = names_outputs if names_outputs is not None else self._model_vars.keys()

        for idx, var in enumerate(iterator):
            if var in self._model_vars:
                refs.append(self._model_vars[var]["ref"])
                names.append(var)
                self._output_map["real"].append(self._model_vars[var]["type"] == "real")
                self._output_map["int"].append(self._model_vars[var]["type"] == "int")
                self._output_map["bool"].append(self._model_vars[var]["type"] == "bool")
            else:
                log.warning(
                    f"Output variable '{var}' couldn't be found in FMU model description. Entry will be ignored."
                )
        self._outputs["names"] = names
        self._outputs["refs"] = refs
        self._outputs["real"] = list(it.compress(refs, self._output_map["real"]))
        self._outputs["int"] = list(it.compress(refs, self._output_map["int"]))
        self._outputs["bool"] = list(it.compress(refs, self._output_map["bool"]))

        #: Directory where the FMU will be extracted.
        self._unzipdir: Path = extract(fmu_path)

        try:
            #: Instance of the FMU Slave object.
            self.fmu: FMU2Slave = FMU2Slave(
                guid=self.model_description.guid,
                unzipDirectory=self._unzipdir,
                modelIdentifier=self.model_description.coSimulation.modelIdentifier,
                instanceName="FMUsimulator_" + str(_id),
            )
        except Exception:
            compile_platform_binary(self.fmu_path)
            self.fmu = FMU2Slave(
                guid=self.model_description.guid,
                unzipDirectory=self._unzipdir,
                modelIdentifier=self.model_description.coSimulation.modelIdentifier,
                instanceName="FMUsimulator_" + str(_id),
            )

        # initialize
        self.fmu.instantiate(visible=False, callbacks=None, loggingOn=False)
        self.fmu.setupExperiment(startTime=self.start_time)
        self.fmu.enterInitializationMode()

        init_values = {} if init_values is None else init_values
        apply_start_values(self.fmu, self.model_description, start_values=init_values)

        self.fmu.exitInitializationMode()
        #: Current simulation time.
        self.time = self.start_time

        # Initialize some other parameters used to switch functionality of class methods.
        #: Return dictionaries from the step and get_values functions instead of lists.
        self._return_dict: bool = False
        if return_type is None:
            self._return_dict = False if names_inputs is not None and names_outputs is not None else True
        else:
            self._return_dict = False if return_type == "list" else True

    @property
    def input_vars(self) -> list[str]:
        """Ordered list of all available input variable names in the FMU."""
        return self._inputs["names"].copy()

    @property
    def output_vars(self) -> list[str]:
        """Ordered list of all available output variable names in the FMU."""
        return self._outputs["names"].copy()

    def read_values(self, names: Sequence[str] | None = None) -> dict[str | int, Any] | list:
        """Return current values of the simulation without advancing a simulation step or the simulation time.

        :param names: Sequence of values to read from the FMU. If this is None (default), all available values will be
                      read.
        """
        # Find value references and names for the variables that should be read from the FMU
        if names is None:
            refs = self._outputs["refs"]
            vars_ = self._outputs["names"]
        else:
            refs = []
            vars_ = []
            for var in names:
                try:
                    refs.append(self._model_vars[var]["ref"])
                    vars_.append(var)
                except KeyError:
                    raise KeyError(f"Specified an input value for a variable which is not available in the FMU: {var}")

        # Get values from the FMU and convert to specified output format (dict or list)
        output_values = self.fmu.getReal(refs)
        output = dict(zip(vars_, output_values)) if self._return_dict else output_values

        return output

    def set_values(self, values: Sequence[Number | bool] | Mapping[str, Number | bool]) -> None:
        """Set values of simulation variables without advancing a simulation step or the simulation time.

        :param values: Values that should be pushed to the FMU. Names of the input_values must correspond
                       to variables in the FMU. If passing as a Sequence, make sure the order corresponds to
                       the order of the input_vars property.
        """

        vals: dict[str, list[Number | bool]] = {"real": [], "int": [], "bool": []}
        refs: dict[str, list[str]] = {"real": [], "int": [], "bool": []}
        if isinstance(values, Mapping):
            for var, val in values.items():
                try:
                    refs[self._model_vars[var]["type"]].append(self._model_vars[var]["ref"])
                    vals[self._model_vars[var]["type"]].append(val)
                except KeyError:
                    raise KeyError(f"Specified an input value for a variable which is not available in the FMU: {var}")
        else:
            if len(values) != len(self._inputs["refs"]):
                raise AttributeError(
                    f"Length of value list ({len(values)}) must be equal to length of input_vars "
                    f"property ({len(self._inputs['refs'])})"
                )
            refs = {
                "real": self._inputs["real"],
                "int": self._inputs["int"],
                "bool": self._inputs["bool"],
            }

            vals = {
                "real": list(it.compress(values, self._input_map["real"])),
                "int": list(it.compress(values, self._input_map["int"])),
                "bool": list(it.compress(values, self._input_map["bool"])),
            }

        if len(refs["real"]) > 0:
            self.fmu.setReal(refs["real"], vals["real"])
        if len(refs["int"]) > 0:
            self.fmu.setInteger(refs["int"], vals["int"])
        if len(refs["bool"]) > 0:
            self.fmu.setBoolean(refs["bool"], vals["bool"])

    def step(
        self,
        input_values: Sequence[Number | bool] | Mapping[str, Number | bool] | None = None,
        output_names: Sequence[str] | None = None,
        advance_time: bool = True,
        nr_substeps: int | None = None,
    ) -> dict[str | int, Any] | list[Any]:
        """Simulate next time step in the FMU with defined input values and output values.

        :param input_values: Current values that should be pushed to the FMU. Names of the input_values must correspond
                             to variables in the FMU. If passing as a Sequence, make sure the order corresponds to
                             the order of the input_vars property.
        :param advance_time: Decide if the FMUsimulator should add one timestep to the simulation time or not.
                                  This can be deactivated, if you just want to look at the result of a simulation step
                                  beforehand, without actually advancing simulation time.
        :param nr_substeps: if simulation steps are divided into substeps, this value will let the simulator know
                                that no time violation warning is necessary.
        :return: Resulting input and output values from the FMU with the keys named corresponding to the variables
                 in the FMU.
        """
        if input_values is not None:
            self.set_values(input_values)

        # put out warning for time limit violation, if self.time + self.step_size > self.stop_time + full step size
        if self.time + self.step_size > self.stop_time + (int(nr_substeps) if nr_substeps else 1) * self.step_size:
            log.warning(
                f"Simulation time {self.time + self.step_size} s exceeds specified stop time of "
                f"{self.stop_time} s. Proceed with care, simulation may become inaccurate."
            )

        # push input values to the FMU and do one timestep, doStep performs a step of certain size
        self.fmu.doStep(currentCommunicationPoint=self.time, communicationStepSize=self.step_size)

        # advance time
        if advance_time:
            self.time += self.step_size  # advance the time

        output = self.read_values(output_names)
        return output

    @classmethod
    def simulate(
        cls,
        fmu_path: Path,
        start_time: TimeStep = 0,
        stop_time: TimeStep = 1,
        step_size: TimeStep = 1,
        init_values: Mapping[str, float] | None = None,
    ) -> np.ndarray:
        """Instantiate a simulator with the specified FMU, perform simulation and return results.

        :param fmu_path: Path to the FMU file.
        :param start_time: Simulation start time in seconds.
        :param float stop_time: Simulation stop time in seconds.
        :param step_size: simulation step size in seconds.
        :param init_values: Starting values for parameters that should be pushed to the FMU with names corresponding to
                            variables in the FMU.
        """

        simulator = cls(0, fmu_path, start_time, stop_time, step_size, init_values=init_values)

        dt = np.dtype([(name, float) for name in simulator.read_values()])
        result = np.rec.array(
            None, shape=((simulator.stop_time - simulator.start_time) // simulator.step_size + 1,), dtype=dt
        )

        step = 0
        while simulator.time <= simulator.stop_time:
            step_result = simulator.step()
            for name in result.dtype.names:
                result[step][name] = step_result[name]

            step += 1

        return result

    def reset(self, init_values: Mapping[str, float] | None = None) -> None:
        """Reset FMU to specified initial condition.

        :param init_values: Values for initialization.
        """
        self.time = self.start_time
        self.fmu.reset()
        self.fmu.setupExperiment(startTime=self.start_time)
        self.fmu.enterInitializationMode()

        apply_start_values(self.fmu, self.model_description, start_values=init_values)

        self.fmu.exitInitializationMode()

    def close(self) -> None:
        """Close the FMU and tidy up the unzipped files."""
        self.fmu.terminate()
        self.fmu.freeInstance()
        shutil.rmtree(self._unzipdir)  # clean up unzipped files


class FMU2MESlave(FMU2Model):
    """Helper class for simulation of FMU2 FMUs. This is as wrapper for FMU2Model.
    It can be used to wrap model exchange FMUs such that they can be simulated similar to a co-simulation FMU. This
    is especially helpful for testing model exchange FMUs.

    It exposes an interface that emulates part of the original FMU2Slave class from fmpy.
    """

    # Define some constants that might be needed according to the FMI Standard
    fmi2True: int = 1  # noqa: N815
    fmi2False: int = 0  # noqa: N815

    fmi2OK: int = 0  # noqa: N815
    fmi2Warning: int = 1  # noqa: N815
    fmi2Discard: int = 2  # noqa: N815
    fmi2Error: int = 3  # noqa: N815
    fmi2Fatal: int = 4  # noqa: N815
    fmi2Pending: int = 5  # noqa: N815

    def __init__(self, **kwargs: Any) -> None:
        r"""Initialize the FMU2Slave object. See also the fmyp documentation :py:class:`fmpy.fmi2.FMU2Model`.

        :param Any \**kwargs: Accepts any parameters that fmpy.FMU2Model accepts.
        """

        super().__init__(**kwargs)
        self._model_description: ModelDescription = read_model_description(kwargs["unzipDirectory"])
        self._solver: CVodeSolver
        self._tolerance: float = 0.0
        self._stop_time: float = 0.0
        self._start_time: float = 0.0

    def setupExperiment(  # noqa: N802
        self,
        tolerance: float | None = None,  # noqa:N803
        startTime: float = 0.0,  # noqa:N803
        stopTime: float | None = None,  # noqa:N803
        **kwargs: Any,
    ) -> int:
        """Experiment setup and storage of required values.

        .. see also::
            fmpy.fmi2.FMU2Model.setupExperiment

        :param tolerance: Solver tolerance, default value is 1e-5.
        :param startTime: Starting time for the experiment.
        :param stopTime: Ending time for the experiment.
        :param kwargs: Other keyword arguments that might be required for FMU2Model.setupExperiment in the future.
        :return: FMI2 return value.
        """

        self._tolerance = 1e-5 if tolerance is None else tolerance
        self._stop_time = 0.0 if stopTime is None else stopTime
        self._start_time = startTime

        kwargs["tolerance"] = self._tolerance
        kwargs["stopTime"] = self._stop_time
        kwargs["startTime"] = self._start_time

        return super().setupExperiment(**kwargs)

    def exitInitializationMode(self, **kwargs: Any) -> int:  # noqa: N802
        """Exit the initialization mode and set up the cvode solver.

        See also: :py:class:`fmpy.fmi2.FMU2Model.exitInitializationMode`

        :param kwargs: Keyword arguments accepted by FMU2Model.exitInitializationMode.
        :return: FMI2 return value.
        """

        ret = super().exitInitializationMode(**kwargs)

        # Collect discrete states from FMU
        self.eventInfo.newDiscreteStatesNeeded = self.fmi2true
        self.eventInfo.terminateSimulation = self.fmi2false

        while (
            self.eventInfo.newDiscreteStatesNeeded == self.fmi2true
            and self.eventInfo.terminateSimulation == self.fmi2false
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

    def doStep(  # noqa: N802
        self,
        currentCommunicationPoint: float,  # noqa: N803
        communicationStepSize: float,  # noqa: N803
        noSetFMUStatePriorToCurrentPoint: int | None = None,  # noqa: N803
    ) -> int:
        """Perform a simulation step. Advance simulation from *currentCommunicationPoint* by *communicationStepSize*.

        Also refer to the FMI2 Standard documentation.

        :param currentCommunicationPoint: Current time stamp (starting point for simulation step).
        :param communicationStepSize: Time step size.
        :param noSetFMUStatePriorToCurrentPoint: Determine whether a reset before *currentCommunicationPoint* is
            possible. Must be either fmi2True or fmi2False.
        :return: FMU2 return value.
        """
        time = currentCommunicationPoint
        step_size = communicationStepSize

        # Perform a solver step and reset the FMU Model time.
        _, time = self._solver.step(time, time + step_size)
        self.setTime(time)
        # Check for events that might have occurred during the step
        step_event, _ = self.completedIntegratorStep()

        return self.fmi2ok
