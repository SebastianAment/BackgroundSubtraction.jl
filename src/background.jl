# data matrix A consists of columns of spectrograms
# k is the number of components of the background model
# x are the index values for each spectrogram (i.e. column of A)
# e.g.: if A[:, 1] is an XRD pattern, x should be the q-values associated with the pattern
function mcbl(A::AbstractMatrix, k::Int, x::AbstractVector, l::Real;
                                        minres::Real = 1e-2, nsigma::Real = 2,
                                        maxiter::Int = 32, minnpeak::Int = 1)
    project_u! = smooth_projection(x, l, k)
    mcbl(A, k, project_u!, minres = minres, nsigma = nsigma,
                        maxiter = maxiter, minnpeak = minnpeak)
end

# fallback for vector input
function mcbl(A::AbstractVector, x::AbstractVector, l::Real)
    project_u! = smooth_projection(x, l)
    mcbl(A, project_u!)
end

# A, k same as above
# project_u! is the projection of the background components into a RKHS
function mcbl(A::AbstractMatrix, k::Int, project_u!;
                                minres::Real, nsigma::Real,
                                maxiter::Int, minnpeak::Int)
    n, m = size(A)
    if k ≥ n || k ≥ m
        throw(DimensionMismatch("k = $k exceeds a matrix dimension: size(A) = ($n, $m)"))
    end
    L = LowRank(rand, n, k, m)
    measurement = copy(A)
    background = similar(A)
    function projection!(background, measurement)
        pals!(L, measurement, project_u!, maxiter = 32)
        mul!(background, L.U, L.V)
    end
    projected_background!(background, measurement, projection!,
                                        minres = minres, nsigma = nsigma,
                                        maxiter = maxiter, minnpeak = minnpeak)
end

function mcbl(A::AbstractVector, project_u!)
    measurement = copy(A)
    background = similar(A)
    function projection!(background, measurement)
        copyto!(background, measurement)
        background = project_u!(background)
    end
    projected_background!(background, measurement, projection!)
end

# data matrix measurement consists of columns of spectrograms,
# WARNING: overwrites measurement with background estimate
# minnpeak is the minimum number of positive outliers below which the algorithm terminates
function projected_background!(background::AbstractArray, measurement::AbstractArray,
                                projection!; minres::Real = 1e-2, nsigma::Real = 2,
                                            maxiter::Int = 32, minnpeak::Int = 1)
    ispeak = BitArray(undef, size(background))
    δ = similar(background)
    for i in 1:maxiter
        projection!(background, measurement) # the projected measurement is an approximation to the background
        @. δ = measurement - background
        σ = std(δ)
        σ > minres || break
        @. ispeak = δ > nsigma * σ # TODO: local smoothing
        sum(ispeak) > minnpeak || break
        @. measurement[ispeak] = background[ispeak] # overwrite large positive outliers with background model
    end
    background
end

# tol is the threshold for detecting rank-deficiency
# ncomp is the number of components of the background model, used for pre-allocation
function smooth_projection(x::AbstractVector, l::Real, ncomp::Int = 0; tol::Real = 1e-6)
    k = Kernel.Lengthscale(Kernel.EQ(), l) # forms RKHS of background signal
    P = Projection(Kernel.gramian(k, x), tol)
    QXsize = ncomp == 0 ? size(P.Q, 2) : (size(P.Q, 2), ncomp)
    QX = zeros(eltype(x), QXsize) # pre-allocation for temporary needed for Projection
    return function project_u!(X) # projects matrix U in low rank factorization U*V
        mul!(X, P, X, QX)
    end
end
