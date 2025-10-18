using LearnAPI
using Preferences

@testset "default_verbosity" begin
    @test LearnAPI.default_verbosity() == LearnAPI.VERBOSITY
    @test_logs(
        (:info, LearnAPI.INFO_VERBOSITY_IS(LearnAPI.default_verbosity())),
        (:info, LearnAPI.INFO_VERBOSITY_WILL_BE(3)),
        LearnAPI.default_verbosity(3),
    )
    @test load_preference(LearnAPI, "verbosity") == 3

    # restore active preference:
    @test_logs(
        (:info, LearnAPI.INFO_VERBOSITY_IS(LearnAPI.default_verbosity())),
        (:info, LearnAPI.INFO_VERBOSITY_WILL_BE(LearnAPI.VERBOSITY)),
        LearnAPI.default_verbosity(LearnAPI.VERBOSITY),
    )
end

true
