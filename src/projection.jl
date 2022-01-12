########################### Projection Matrix ##################################
# stands for A*(AF\y) = A*inverse(A'A)*(A'y) = A*pseudoinverse(A)
struct Projection{T, QT<:AbstractMatOrFac{T}} <: Factorization{T}
    Q::QT # orthogonal matrix
    function Projection(Q::AbstractMatOrFac; check::Bool = true)
        (check && Q'Q ≈ I(size(Q, 2))) || throw("Input matrix Q not orthogonal. Call projection(Q) instead.")
        new{eltype(Q), typeof(Q)}(Q)
    end
end

function projection(A::AbstractMatrix; tol::Real = eps(eltype(A)))
    F = qr(A, ColumnNorm())
    r = rank(F, tol)
    Q = Matrix(F.Q)[:, 1:r]
    Projection(Q)
end

function LinearAlgebra.rank(F::QRPivoted, tol::Real = eps(eltype(F)))
    ind = size(F.R, 2)
    for (i, di) in enumerate(diagind(F.R))
        if abs(F.R[di]) < tol
            ind = i-1
            break
        end
    end
    return ind
end

(P::Projection)(x::AbstractVecOrMatOrFac) = P.Q * (P.Q' * x)

# Qx is memory pre-allocation which can be passed optionally
function LinearAlgebra.mul!(y::AbstractVecOrMat, P::Projection, x::AbstractVecOrMat,
                                Qx::AbstractVecOrMat, α::Real = 1, β::Real = 0)
    _mul_helper!(y, P, x, Qx, α, β)
end

@inline function _mul_helper!(y, P::Projection, x, Qx, α = 1, β = 0)
    mul!(Qx, P.Q', x)
    mul!(y, P.Q, Qx, α, β)
end

function LinearAlgebra.mul!(y::AbstractVecOrMat, P::Projection, x::AbstractVecOrMat,
                                                        α::Real = 1, β::Real = 0)
    Qx = P.Q' * x
    mul!(y, P.Q, Qx, α, β)
end


function LinearAlgebra.mul!(y::AbstractVecOrMat, x::AbstractVecOrMat, P::Projection,
                                                        α::Real = 1, β::Real = 0)
    mul!(y', P', x', α, β)
end

Base.size(P::Projection, k::Integer) = 0 < k ≤ 2 ? size(P.Q, 1) : 1
Base.size(P::Projection) = (size(P, 1), size(P, 2))
Base.eltype(P::Projection{T}) where {T} = T

# properties
LinearAlgebra.Matrix(P::Projection) = P.Q*P.Q'
LinearAlgebra.adjoint(P::Projection) = P
LinearAlgebra.transpose(P::Projection) = P
Base.:^(P::Projection, n::Integer) = P
function Base.literal_pow(::typeof(^), P::Projection, ::Val{N}) where N
    N > 0 ? P : error("Projection P is not invertible")
end
Base.:*(P::Projection, x::AbstractVecOrMat) = P(x)
Base.:*(x::AbstractVecOrMat, P::Projection) = P(x')'
