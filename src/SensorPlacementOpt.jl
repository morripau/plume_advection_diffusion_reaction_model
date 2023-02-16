module SensorPlacementOpt

    using CairoMakie, LinearAlgebra

    #=
    mutable struct SensingGridSpace
        V::Vector{Tuple{Float64, Float64}}
        S::Vector{Tuple{Float64, Float64}}
        U::Vector{Tuple{Float64, Float64}}
    end
    =#

    ########################################################################################
    # MUTUAL INFORMATION
    ########################################################################################

    """
    returns the optimal set, A, of sensor placement locations.
    n_place - the number of sensors to place
    V - the complete grid 
    (note V = S ∪ U)
    S - the set of locations in the grid where sensors can be placed
    (note S ⊆ V)
    """
    function opt_placement_mi(n_place::Int, V::Vector{Tuple{Float64, Float64}}, S::Vector{Tuple{Float64, Float64}})
        A = [] # placed sensors
        #n_place = 10 # k in the paper
        for s = 1:n_place
            # all locations without sensors yet.
            candidate_locs = build_current_candidates(A, S) # S \ A

            δs = zeros(length(candidate_locs))
            # loop over all candidate sensors
            for (c, y) in enumerate(candidate_locs)
                # set of all locations, outside of union(A, y), 
                # we care abt predictions here. want them to be certain.
                Ā = [(i, j) for (i, j) in V if (! ((i, j) in A)) && (i, j) != y]
                
                Σ_yA = [k(y, a) for a in A]
                Σ_AA = [k(aᵢ, aⱼ) for aᵢ in A, aⱼ in A]

                Σ_yĀ = [k(y, ā) for ā in Ā]
                Σ_ĀĀ = [k(āᵢ, āⱼ) for āᵢ in Ā, āⱼ in Ā]
            
                σ_y = k(y, y)

                top = σ_y - Σ_yA' * inv(Σ_AA) * Σ_yA
                bot = σ_y - Σ_yĀ' * inv(Σ_ĀĀ) * Σ_yĀ

                δs[c] = top / bot
            end
            push!(A, candidate_locs[argmax(δs)])
        end
        return A
    end

    """
    returns locations where a sensor is not yet placed
    """
    function build_current_candidates(A, S)
        return [(i, j) for (i, j) in S if ! ((i, j) in A)]
    end


    ########################################################################################
    # KERNELS
    ########################################################################################

    """
    radial basis function (RBF) kernel
    """
    function k(i::Tuple{Float64, Float64}, j::Tuple{Float64, Float64}; γ=2.0)
        return exp(-sum((i .- j) .^ 2) / γ^2)
    end

end