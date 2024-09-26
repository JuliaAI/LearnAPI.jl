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

@trait(
    Selector,
    constructor = Selector,
    functions = (
        :(LearnAPI.fit),
        :(LearnAPI.algorithm),
        :(LearnAPI.minimize),
        :(LearnAPI.obs),
        :(LearnAPI.transform),
    ),
)

@testset "test a static transformer" begin
    algorithm = Selector(names=[:x, :w])
    X = DataFrames.DataFrame(rand(3, 4), [:x, :y, :z, :w])
    model = fit(algorithm) # no data arguments!
    @test LearnAPI.algorithm(model) == algorithm
    W = transform(model, X)
    @test W == DataFrames.DataFrame(Tables.matrix(X)[:,[1,4]], [:x, :w])
    @test W == transform(algorithm, X)
end


# # FEATURE SELECTOR THAT REPORTS BYPRODUCTS OF SELECTION PROCESS

# This a variation of `Selector` above that stores the names of rejected features in the
# model object, for inspection by an accessor function called `rejected`. Since
# `transform(model, X)` mutates `model` in this case, we must overload the
# `predict_or_transform_mutates` trait.

struct Selector2
    names::Vector{Symbol}
end
Selector2(; names=Symbol[]) =  Selector2(names) # LearnAPI.constructor defined later

mutable struct Selector2Fit
    algorithm::Selector2
    rejected::Vector{Symbol}
    Selector2Fit(algorithm) = new(algorithm)
end
LearnAPI.algorithm(model::Selector2Fit) = model.algorithm
rejected(model::Selector2Fit) = model.rejected

# Here we are wrapping `algorithm` with a place-holder for the `rejected` feature names.
LearnAPI.fit(algorithm::Selector2; verbosity=1) = Selector2Fit(algorithm)

# output the filtered table and add `rejected` field to model (mutatated!)
function LearnAPI.transform(model::Selector2Fit, X)
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
function LearnAPI.transform(algorithm::Selector2, X)
    model = fit(algorithm)
    transform(model, X)
end

@trait(
    Selector2,
    constructor = Selector2,
    predict_or_transform_mutates = true,
    functions = (
        :(LearnAPI.fit),
        :(LearnAPI.algorithm),
        :(LearnAPI.minimize),
        :(LearnAPI.obs),
        :(LearnAPI.transform),
        :(MyPkg.rejected), # accessor function not owned by LearnAPI.jl,
    )
)

@testset "test a variation that reports byproducts" begin
    algorithm = Selector2(names=[:x, :w])
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
