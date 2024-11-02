using Test

@test LearnAPI.default_verbosity() ==1
LearnAPI.default_verbosity(42)
@test LearnAPI.default_verbosity() == 42

true
