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
    "If you have not installe Julia, download from "
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
    jl = import_jl_file(file)
    return jl


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
    elif not found or not file:
        raise ImportError(f"Could not find the specified julia file. Looking for {pathstr}")

    return file


def install_julia() -> None:
    """Checks if Julia language is available in the system and install and configure pyjulia.
    Also install ju_extensions in Julia environmnent.
    """
    if which("julia") is None:
        raise Exception(JULIA_NOT_FOUND_MSG)

    try:
        import julia  # noqa: I900
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "julia"])

        import julia  # noqa: I900

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
    """Check if PyJulia package is available in python interpreter.

    :returns: True if is installed return Error if not
    """
    try:
        import julia  # noqa: I900 F401
    except ModuleNotFoundError:
        raise ImportError(
            "Could not find the julia package. Please run the command: install-julia."
            "Inside the python virtual environment where eta-utility is installed."
        )
    return True
