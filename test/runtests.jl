using Test

@testset "tools.jl" begin
    include("tools.jl")
end

# # INTEGRATION TESTS

@testset "regression" begin
    include("integration/regression.jl")
end

# @testset "classification" begin
#     include("integration/classification.jl")
# end

# @testset "clustering" begin
#     include("integration/clustering.jl")
# end

# @testset "gradient_descent" begin
#     include("integration/gradient_descent.jl")
# end

# @testset "iterative_algorithms" begin
#     include("integration/iterative_algorithms.jl")
# end

# @testset "incremental_algorithms" begin
#     include("integration/incremental_algorithms.jl")
# end

# @testset "dimension_reduction" begin
#     include("integration/dimension_reduction.jl")
# end

# @testset "encoders" begin
#     include("integration/encoders.jl")
# end

@testset "static_algorithms" begin
    include("integration/static_algorithms.jl")
end

# @testset "missing_value_imputation" begin
#     include("integration/missing_value_imputation.jl")
# end

# @testset "ensemble_algorithms" begin
#     include("integration/ensemble_algorithms.jl")
# end

# @testset "wrappers" begin
#     include("integration/wrappers.jl")
# end

# @testset "time_series_forecasting" begin
#     include("integration/time_series_forecasting.jl")
# end

# @testset "time_series_classification" begin
#     include("integration/time_series_classification.jl")
# end

# @testset "survival_analysis" begin
#     include("integration/survival_analysis.jl")
# end

# @testset "distribution_fitters" begin
#     include("integration/distribution_fitters.jl")
# end

# @testset "Bayesian_algorithms" begin
#     include("integration/Bayesian_algorithms.jl")
# end

# @testset "outlier_detection" begin
#     include("integration/outlier_detection.jl")
# end

# @testset "collaborative_filtering" begin
#     include("integration/collaborative_filtering.jl")
# end

# @testset "text_analysis" begin
#     include("integration/text_analysis.jl")
# end

# @testset "audio_analysis" begin
#     include("integration/audio_analysis.jl")
# end

# @testset "natural_language_processing" begin
#     include("integration/natural_language_processing.jl")
# end

# @testset "image_processing" begin
#     include("integration/image_processing.jl")
# end
