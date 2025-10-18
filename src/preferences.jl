const VERBOSITY = @load_preference "verbosity" 1

INFO_VERBOSITY_IS(verbosity) =
    "Currently the baseline verbosity is $verbosity. "
INFO_VERBOSITY_WILL_BE(verbosity) =
    "After restarting Julia, this will be changed to $verbosity. "

"""
    LearnAPI.default_verbosity()

Return the default verbosity for training LearnAPI learners.

The value is determined at compile time by a Preferences.jl-style preference, with key
"verbosity".

"""
default_verbosity() = VERBOSITY

"""
    LearnAPI.default_verbosity(level)

Set the default verbosity for training LearnAPI learners to `level`. Changes do not take
effect until the next Julia session.

"""
function default_verbosity(level)
    @info INFO_VERBOSITY_IS(VERBOSITY)
    @set_preferences! "verbosity" => level
    @info INFO_VERBOSITY_WILL_BE(level)
end
