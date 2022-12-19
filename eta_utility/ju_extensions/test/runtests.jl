using PyCall

import Base.Filesystem: normpath, joinpath, splitdir
import Base: source_path

# Make sure that the correct python library folders are being used for the tests
# (site packages and eta_utility package path).
pushfirst!(
    PyVector(pyimport("sys")."path"),
    normpath(joinpath(splitdir(PyCall.pyprogramname)[1], "../Lib/site-packages")),
)
pushfirst!(PyVector(pyimport("sys")."path"), normpath(joinpath(splitdir(source_path())[1], "../../../")))

include("etax/agents/NSGA2.jl")
