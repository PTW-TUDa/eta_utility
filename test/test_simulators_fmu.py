import os.path
from test.config_tests import FMU_FILE

from pytest import approx, fixture

from eta_utility.simulators import FMUSimulator


class TestFMUSimulator:
    @fixture()
    def seq_simulator(self, monkeypatch):
        """Legacy initialization required all values and would expect the simulator to return lists. A simulator
        which does this is initialized here."""
        init_values = {"u": 0}

        # monkeypatch.setattr(eta_utility.simulators.fmu, "read_model_description", read_model_description)

        simulator = FMUSimulator(
            0,
            fmu_path=FMU_FILE,
            start_time=0,
            stop_time=100,
            step_size=1,
            names_inputs=["u"],
            names_outputs=["s", "v", "a"],
            init_values=init_values,
        )
        return simulator

    @fixture(scope="class", autouse=False)
    def map_simulator(self):
        """New format initialization also allows for the simulator to be initialized with just the fmu_path."""

        simulator = FMUSimulator(0, fmu_path=FMU_FILE)
        return simulator

    def test_attributes(self, seq_simulator):
        """Check whether most important attributes are present"""
        assert os.path.isdir(seq_simulator._unzipdir)

        assert hasattr(seq_simulator, "step")
        assert hasattr(seq_simulator, "reset")
        assert hasattr(seq_simulator, "close")

    def test_step_sequence_input(self, seq_simulator):
        """Test stepping function with the sequence input and output formats"""
        input_values = [0.5]
        s, v, a = seq_simulator.step(input_values)

        assert s == approx(0.768, 0.01)
        assert v == approx(0.569, 0.01)
        assert a == approx(-1.627, 0.01)

        s, v, a = seq_simulator.step(input_values)
        assert s == approx(0.550, 0.01)
        assert v == approx(-0.682, 0.01)
        assert a == approx(0.089, 0.01)

    def test_set_read_map_values(self, map_simulator):
        """Test setting and reading a value from a mapping"""
        init_values = {"u": 0}
        map_simulator.set_values(init_values)
        output = map_simulator.read_values(["u"])
        assert output["u"] == 0.0

    def test_step_map_input(self, map_simulator):
        """Test stepping function with the mapping input and output formats"""
        input_values = {"u": 0.5}
        output_names = ["s", "v", "a"]

        output = map_simulator.step(input_values, output_names)
        s = output["s"]
        v = output["v"]
        a = output["a"]
        assert s == approx(0.768, 0.01)
        assert v == approx(0.569, 0.01)
        assert a == approx(-1.627, 0.01)

        output = map_simulator.step(input_values)
        s = output["s"]
        v = output["v"]
        a = output["a"]
        assert s == approx(0.550, 0.01)
        assert v == approx(-0.682, 0.01)
        assert a == approx(0.089, 0.01)

    def test_FMUSimulator_reset(self, seq_simulator):
        """Test resetting the simulator"""
        seq_simulator.reset({"u": 0})

        assert seq_simulator.time == 0

    def test_FMUSimulator_close(self, seq_simulator):
        """Test closing the simulator object"""
        seq_simulator.close()

        assert not os.path.isdir(seq_simulator._unzipdir)
