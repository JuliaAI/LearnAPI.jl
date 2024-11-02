using Test
using LearnAPI

struct Cherry end

LearnAPI.fit(learner::Cherry, data; verbosity=1) = Ref(learner)
LearnAPI.learner(model::Base.RefValue{Cherry}) = model[]
LearnAPI.predict(model::Base.RefValue{Cherry}, ::Point, x) = 2x
@trait Cherry kinds_of_proxy=(Point(),)

struct Ripe end

LearnAPI.fit(learner::Ripe, data; verbosity=1) = Ref(learner)
LearnAPI.learner(model::Base.RefValue{Ripe}) = model[]
LearnAPI.predict(model::Base.RefValue{Ripe}, ::Distribution) = "a distribution"
LearnAPI.features(::Ripe, data) = nothing
@trait Ripe kinds_of_proxy=(Distribution(),)

@testset "`predict` with no kind of proxy specified" begin
    model = fit(Cherry(), "junk")
    @test predict(model, 42) == 84

    model = fit(Ripe(), "junk")
    @test predict(model) == "a distribution"
end

true
