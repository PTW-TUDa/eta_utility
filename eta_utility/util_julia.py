from __future__ import annotations

import importlib
import inspect
import os
import pathlib
import subprocess
import sys
from shutil import which
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from types import ModuleType

    from eta_utility.type_hints import Path

# Set environment variable to determine the correct python environment to use when calling back into
# python from julia
os.environ["PYCALL_JL_RUNTIME_PYTHON"] = sys.executable

JULIA_NOT_FOUND_MSG = (
    "Julia executable cannot be found. "
    "If you have installed Julia, make "
    "sure Julia executable is in the system path. "
    "If you have not installed Julia, download from "
    "https://julialang.org/downloads/ and install it. "
)


def import_jl_file(filename: Path) -> ModuleType:
    check_julia_package()

    _filename = pathlib.Path(filename) if not isinstance(filename, pathlib.Path) else filename
    jl = importlib.import_module("julia.Main")

    jl.include(_filename.absolute().as_posix())
    return jl


def import_jl(importstr: str) -> ModuleType:
    """Import a julia file into the main namespace. The syntax is equivalent to python import strings. If the import
    string starts with a '.', the import path is interpreted as relative to the file calling this function. If the
    import string is absolute, it will use the python sys.path list to look for the file.

    The function also makes sure that julia.Main is imported and returns a handle to the module. This way, the
    imported julia file can be used right away.

    :param importstr: Path to the imported julia package. If the path starts with a '.' this will be relative to the
        file it is specified in. Otherwise, this will look through python import Path.
    """
    check_julia_package()

    file = importstr_to_path(importstr, _stack=2)
    return import_jl_file(file)


def importstr_to_path(importstr: str, _stack: int = 1) -> pathlib.Path:
    """Converts an import string into a python path. The syntax is equivalent to python import strings. If the
    import string starts with a '.', the import path is interpreted as relative to the file calling this function.
    If the import string is absolute, it will use the python sys.path list to look for the file.

    :param importstr: Path to the imported julia package (python import string). If the path starts with a '.' this
        will be relative to the file it is specified in. Otherwise, this will look through the python import paths.
    """
    if len(importstr) > 2 and importstr[0] == "." and importstr[1] == ".":
        pathstr = f"..{importstr[2:].replace('.', '/')}.jl"
        relative = True
    elif len(importstr) > 1 and importstr[0] == ".":
        pathstr = f"{importstr[1:].replace('.', '/')}.jl"
        relative = True
    else:
        pathstr = f"{importstr.replace('.', '/')}.jl"
        relative = False

    file = None
    found = False
    if relative:
        file = pathlib.Path(inspect.stack()[_stack].filename).parent / pathstr
        if file.is_file():
            found = True
    else:
        for path in sys.path:
            file = pathlib.Path(path) / pathstr
            if file.is_file():
                found = True
                break

    if not found and relative and file:
        raise ImportError(f"Could not find the specified julia file. Looking for {file}")
    if not found or not file:
        raise ImportError(f"Could not find the specified julia file. Looking for {pathstr}")

    return file


def update_agent() -> None:
    """Upadtes the NSGA2 agent model file"""
    import tempfile

    from test.test_etax.test_agents import TestNSGA2

    cls = TestNSGA2()
    cls.create_stored_agent_file("test/resources/agents/", tempfile.TemporaryDirectory().name)


def install_julia() -> None:
    """Checks if Julia language is available in the system and install and configure pyjulia.
    Also install ju_extensions in Julia environmnent.
    """
    if which("julia") is None:
        raise ImportError(JULIA_NOT_FOUND_MSG)

    try:
        import julia
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "julia"])

        import julia

    # Set environment variable to determine the correct python environment to use when calling back into
    # python from julia
    os.environ["PYCALL_JL_RUNTIME_PYTHON"] = sys.executable
    julia.install()

    subprocess.check_call(
        [
            "julia",
            "-e",
            f"import Pkg; Pkg.develop(path=\"{(pathlib.Path(__file__).parent / 'ju_extensions').as_posix()}\")",
        ]
    )


def check_julia_package() -> bool:
    """Check if everything is available and setup correctly to execute modules depending on eta_utility
    julia extensions. This function raises ImportError if necessary components are missing.

    :returns: True if is installed ImportError if not
    """
    if which("julia") is None:
        raise ImportError(JULIA_NOT_FOUND_MSG)

    try:
        import julia
    except ModuleNotFoundError:
        raise ImportError(
            "Could not find the python julia package. Please run the command: install-julia "
            "inside the python virtual environment where eta-utility is installed."
        ) from None

    try:
        from julia import ju_extensions  # noqa: F401
    except julia.core.UnsupportedPythonError as e:
        raise ImportError(
            "PyCall for Julia is installed for a different python binary than you are currently "
            "using. Please run the command: install-julia inside the python virtual environment "
            "where eta-utility is installed."
        ) from e
    except (ModuleNotFoundError, ImportError, AttributeError) as e:
        raise ImportError(
            "Could not find julia extension module for eta_utility (ju_extensions missing). Please "
            "run the command: install-julia inside the python virtual environment where eta-utility "
            "is installed."
        ) from e

    return True


def julia_extensions_available() -> bool:
    """Check if everything is available and setup correctly to execute modules depending on eta_utility
    julia extensions. This function returns false if necessary components are missing. It does not
    provide any indications what is missing.

    :return: True if julia extensions are correctly installed, false if not.
    """
    try:
        check_julia_package()
    except ImportError:
        return False

    return True
