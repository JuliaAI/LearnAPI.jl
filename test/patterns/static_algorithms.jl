using LearnAPI
using LinearAlgebra
using Tables
import MLUtils
import DataFrames


# # TRANSFORMER TO SELECT SOME FEATURES (COLUMNS) OF A TABLE

# See later for a variation that stores the names of rejected features in the model
# object, for inspection by an accessor function.

struct Selector
    names::Vector{Symbol}
end
Selector(; names=Symbol[]) =  Selector(names) # LearnAPI.constructor defined later

# `fit` consumes no observational data, does no "learning", and just returns a thinly
# wrapped `algorithm` (to distinguish it from the algorithm in dispatch):
LearnAPI.fit(algorithm::Selector; verbosity=1) = Ref(algorithm)
LearnAPI.algorithm(model) = model[]

function LearnAPI.transform(model::Base.RefValue{Selector}, X)
    algorithm = LearnAPI.algorithm(model)
    table = Tables.columntable(X)
    names = Tables.columnnames(table)
    filtered_names = filter(in(algorithm.names), names)
    filtered_columns = (Tables.getcolumn(table, name) for name in filtered_names)
    filtered_table = NamedTuple{filtered_names}((filtered_columns...,))
    return Tables.materializer(X)(filtered_table)
end

# fit and transform in one go:
function LearnAPI.transform(algorithm::Selector, X)
    model = fit(algorithm)
    transform(model, X)
end

# note the necessity of overloading `is_static` (`fit` consumes no data):
@trait(
    Selector,
    constructor = Selector,
    tags = ("feature engineering",),
    is_static = true,
    functions = (
        :(LearnAPI.fit),
        :(LearnAPI.algorithm),
        :(LearnAPI.strip),
        :(LearnAPI.obs),
        :(LearnAPI.transform),
    ),
)

@testset "test a static transformer" begin
    algorithm = Selector(names=[:x, :w])
    X = DataFrames.DataFrame(rand(3, 4), [:x, :y, :z, :w])
    model = fit(algorithm) # no data arguments!
    # if provided, data is ignored:
    @test LearnAPI.algorithm(model) == algorithm
    W = transform(model, X)
    @test W == DataFrames.DataFrame(Tables.matrix(X)[:,[1,4]], [:x, :w])
    @test W == transform(algorithm, X)
end


# # FEATURE SELECTOR THAT REPORTS BYPRODUCTS OF SELECTION PROCESS

# This a variation of `Selector` above that stores the names of rejected features in the
# output of `fit`, for inspection by an accessor function called `rejected`.

struct FancySelector
    names::Vector{Symbol}
end
FancySelector(; names=Symbol[]) =  FancySelector(names) # LearnAPI.constructor defined later

mutable struct FancySelectorFitted
    algorithm::FancySelector
    rejected::Vector{Symbol}
    FancySelectorFitted(algorithm) = new(algorithm)
end
LearnAPI.algorithm(model::FancySelectorFitted) = model.algorithm
rejected(model::FancySelectorFitted) = model.rejected

# Here we are wrapping `algorithm` with a place-holder for the `rejected` feature names.
LearnAPI.fit(algorithm::FancySelector; verbosity=1) = FancySelectorFitted(algorithm)

# output the filtered table and add `rejected` field to model (mutatated!)
function LearnAPI.transform(model::FancySelectorFitted, X)
    table = Tables.columntable(X)
    names = Tables.columnnames(table)
    keep = LearnAPI.algorithm(model).names
    filtered_names = filter(in(keep), names)
    model.rejected = setdiff(names, filtered_names)
    filtered_columns = (Tables.getcolumn(table, name) for name in filtered_names)
    filtered_table = NamedTuple{filtered_names}((filtered_columns...,))
    return Tables.materializer(X)(filtered_table)
end

# fit and transform in one step:
function LearnAPI.transform(algorithm::FancySelector, X)
    model = fit(algorithm)
    transform(model, X)
end

# note the necessity of overloading `is_static` (`fit` consumes no data):
@trait(
    FancySelector,
    constructor = FancySelector,
    is_static = true,
    tags = ("feature engineering",),
    functions = (
        :(LearnAPI.fit),
        :(LearnAPI.algorithm),
        :(LearnAPI.strip),
        :(LearnAPI.obs),
        :(LearnAPI.transform),
        :(MyPkg.rejected), # accessor function not owned by LearnAPI.jl,
    )
)

@testset "test a variation that reports byproducts" begin
    algorithm = FancySelector(names=[:x, :w])
    X = DataFrames.DataFrame(rand(3, 4), [:x, :y, :z, :w])
    model = fit(algorithm) # no data arguments!
    @test !isdefined(model, :reject)
    @test LearnAPI.algorithm(model) == algorithm
    filtered =  DataFrames.DataFrame(Tables.matrix(X)[:,[1,4]], [:x, :w])
    @test transform(model, X) == filtered
    @test transform(algorithm, X) == filtered
    @test rejected(model) == [:y, :z]
end

true
