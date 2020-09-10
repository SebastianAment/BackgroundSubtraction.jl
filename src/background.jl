# TODO: MCBL hyper-parameters in one struct
struct MCBL{T}
    minres::T
    nsigma::T
    maxiter::Int
    minnpeak::Int
end
function MCBL(;minres::Real = 1e-3, nsigma::Real = 2., maxiter::Int = 32, minnpeak::Int = 1)
    MCBL(minres, nsigma, maxiter, minnpeak)
end

# mcbl(A...)

##################################### 1D #######################################
# for vector input, i.e. single spectrogram A
function mcbl(A::AbstractVector, x::AbstractVector, l::Real;
                                        minres::Real = 1e-2, nsigma::Real = 2,
                                        maxiter::Int = 32, minnpeak::Int = 1)
    project_u! = smooth_projection(x, l)
    mcbl(A, project_u!, minres = minres, nsigma = nsigma,
                        maxiter = maxiter, minnpeak = minnpeak)
end

function mcbl(A::AbstractVector, project_u!;
                                        minres::Real = 1e-2, nsigma::Real = 2,
                                        maxiter::Int = 32, minnpeak::Int = 1)
    measurement = copy(A)
    background = similar(A)
    function projection!(background, measurement)
        copyto!(background, measurement)
        background = project_u!(background)
    end
    projected_background!(background, measurement, projection!,
                                        minres = minres, nsigma = nsigma,
                                        maxiter = maxiter, minnpeak = minnpeak)
end

#################################### 2D ########################################
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
        _, _, info = pals!(L, measurement, project_u!, maxiter = 32, min_delta = 1e-1minres)
        mul!(background, L.U, L.V)
    end
    projected_background!(background, measurement, projection!,
                                        minres = minres, nsigma = nsigma,
                                        maxiter = maxiter, minnpeak = minnpeak)
end

# for single image input
# y is second input dimension, either composition coordinates, or 2nd image dimension
# l_y the lengthscale in this dimension
function mcbl(A::AbstractMatrix, x::AbstractVector, l_x::Real,
                                y::AbstractVecOrMat, l_y::Real;
                                minres::Real = 1e-2, nsigma::Real = 2,
                                maxiter::Int = 32, minnpeak::Int = 1)
    measurement = copy(vec(A))
    background = similar(measurement)
    projection! = smooth_projection(x, l_x, y, l_y)
    projected_background!(background, measurement, projection!,
                                        minres = minres, nsigma = nsigma,
                                        maxiter = maxiter, minnpeak = minnpeak)
    return reshape(background, size(A))
end
##################################### 3D #######################################
# data tensor A consists of slices of smooth 2d images, i.e. A[:, :, 1] is a 2D image
function mcbl(A::AbstractArray{<:Real, 3}, k::Int, x::AbstractVector, l_x::Real,
                                                y::AbstractVector, l_y::Real;
                                        minres::Real = 1e-2, nsigma::Real = 2,
                                        maxiter::Int = 32, minnpeak::Int = 1)
    project_u! = smooth_projection(x, l_x, y, l_y)
    A2D = reshape(A, :, size(A, 3)) # convert to matrix where each column is an image
    background = mcbl(A2D, k, project_u!, minres = minres, nsigma = nsigma,
                        maxiter = maxiter, minnpeak = minnpeak)
   return reshape(background, size(A))
end

function mcbl(A::AbstractArray{<:Real, 3}, x::AbstractVecOrMat, l_x::Real,
                                y::AbstractVecOrMat, l_y::Real,
                                z::AbstractVecOrMat, l_z::Real;
                                minres::Real = 1e-2, nsigma::Real = 2,
                                maxiter::Int = 32, minnpeak::Int = 1)
    measurement = copy(vec(A))
    background = similar(measurement)
    projection! = smooth_projection(x, l_x, y, l_y, z, l_z)
    projected_background!(background, measurement, projection!,
                                        minres = minres, nsigma = nsigma,
                                        maxiter = maxiter, minnpeak = minnpeak)
    return reshape(background, size(A))
end

######################### abstract subtraction algorithm #######################
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

############################### smooth projection ##############################
# tol is the threshold for detecting rank-deficiency
# ncomp is the number of components of the background model, used for pre-allocation
function smooth_projection(x::AbstractVector, l::Real, ncomp::Int = 0; tol::Real = 1e-6)
    k = Kernel.Lengthscale(Kernel.EQ(), l) # forms RKHS of background signal
    P = projection(Kernel.gramian(k, x), tol = tol)
    QXsize = ncomp == 0 ? size(P.Q, 2) : (size(P.Q, 2), ncomp)
    QX = zeros(eltype(x), QXsize) # pre-allocation for temporary needed for Projection
    return function project_u!(X) # projects matrix U in low rank factorization U*V
        mul!(X, P, X, QX)
    end
end

# smooth projection for higher dimensional data, i.e. images
# or for spectrograms varying with composition values)
# x is a lazily represented Cartesian grid of x (e.g. q) and y (composition) values
# l_x is length scale of background in x
# l_y is length scale of background in y
function smooth_projection(x::AbstractVecOrMat, l_x::Real,
                           y::AbstractVecOrMat, l_y::Real; tol::Real = 1e-6,
                           k_x = Kernel.EQ(), k_y = Kernel.EQ())
    kxy = Kernel.Lengthscale.((k_x, k_y), (l_x, l_y))
    smooth_projection(kxy, (x, y), tol = tol)
end

# smooth projection for 3D function
function smooth_projection(x::AbstractVecOrMat, l_x::Real,
                           y::AbstractVecOrMat, l_y::Real,
                           z::AbstractVecOrMat, l_z::Real; tol::Real = 1e-6,
                           k_x = Kernel.EQ(), k_y = Kernel.EQ(), k_z = Kernel.EQ())
    kxyz = Kernel.Lengthscale.((k_x, k_y, k_z), (l_x, l_y, l_z))
    smooth_projection(kxyz, (x, y, z), tol = tol)
end
function smooth_projection(k::Tuple, x::Tuple; tol::Real = 1e-6)
    k = Kernel.separable(*, k...)
    K = Kernel.gramian(k, grid(x...))
    K = kronecker(reverse(K.factors)) # why reverse? because definition of Kronecker product is "row-major"
    P = kronecker(A -> projection(A, tol = tol), K)
    projection!(Y, X) = (Y .= P*X) # mul!(background, P, measurement)
    projection!(X) = (X .= P*X)
    return projection!
end
