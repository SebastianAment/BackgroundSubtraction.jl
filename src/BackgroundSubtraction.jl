module BackgroundSubtraction
using LinearAlgebra
# using LinearAlgebraExtensions
# using LinearAlgebraExtensions: LowRank, pals!, projection, grid, AbstractMatOrFac
using LazyInverses
using Statistics
using KroneckerProducts: kronecker
using CovarianceFunctions
using CovarianceFunctions: grid

export mcbl
export kronecker_mcbl

const AbstractMatOrFac{T} = Union{AbstractMatrix{T}, Factorization{T}}
const AbstractVecOrMatOrFac{T} = Union{AbstractVector{T}, AbstractMatOrFac{T}}

include("lowrank.jl")
include("projection.jl")
include("background.jl")

end
