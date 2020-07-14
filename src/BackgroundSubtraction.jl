module BackgroundSubtraction
using LinearAlgebra
using LinearAlgebraExtensions
using LinearAlgebraExtensions: LowRank, pals!, projection, grid, AbstractMatOrFac
using KroneckerProducts: kronecker
using Kernel
using Statistics

export mcbl
export kronecker_mcbl

include("background.jl")

end
