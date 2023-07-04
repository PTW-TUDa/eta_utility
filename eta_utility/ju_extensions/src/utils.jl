using Logging, LoggingExtras

"""
Set ju_extensions logger globally.
Considering python logging level and format convention.

DEBUG = 10; INFO = 20; WARNING = 30; ERROR = 40.

:param level: minimum level value for logging.
:param format:
:return: previous global logger.
"""
function set_logger(level::Int, format::String)
    if format == "simple"
        # Create format logger
        logger = FormatLogger() do io, args
            println(io, "[$(args.level)] $(args.message)")
        end
    elseif format == "logname"
        logger = FormatLogger() do io, args
            println(io, "[$(args._module): $(args.level)] $(args.message)")
        end
    elseif format == "time"
        logger = FormatLogger() do io, args
            println(io, "[$(Dates.format(now(), "yyyy-mm-dd HH:MM:SS")) - $(args._module): $(args.level)] $(args.message)")
        end
    else
        throw(ArgumentError("Unknown logging format '$format'."))
    end

    global_logger(_set_logger(logger, level))
end

"""
Set ju_extensions logger globally.
Considering python logging level and format convention.

DEBUG = 10; INFO = 20; WARNING = 30; ERROR = 40.

:param level: minimum level value for logging.
:return: previous global logger.
"""
set_logger(level::Int) = global_logger(_set_logger(global_logger(), level))

function _set_logger(logger, level::Int)
    level_dict = Dict(10 => Logging.Debug, 20 => Logging.Info, 30 => Logging.Warn, 40 => Logging.Error)
    if !haskey(level_dict, level)
        throw(ArgumentError("Logging level '$level' not defined."))
    end

    filtered_logger = EarlyFilteredLogger(
        log -> log._module != "Base" && parentmodule(log._module) != "Base",
        MinLevelLogger(logger, level_dict[level]),
    )
    return filtered_logger
end
