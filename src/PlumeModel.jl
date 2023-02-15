module SensorPlacementOpt
    using DiffEqOperators, DomainSets, OrdinaryDiffEq, DifferentialEquations, ModelingToolkit, CairoMakie, LinearAlgebra, PlutoUI, Random, StatsBase, Distributions, DataFrames, SpecialFunctions, MethodOfLines, GLMakie




    """
    ****************************************************
    Plume model PDE
    ****************************************************
    """
    
    """
    Generates a 3-D vector of concentration by data[x₁, x₂, t]

    wind_str - increase to push gas left
    diffusivity - increase to increase diffusion of gas
    decay - increase to allow decay
    source_loc - (Float64, Float64) coordinates between 0 and 10
    Δt - time step value
    t_end - run time
    Δx₁ - grid discretization in the x₁ direction
    Δx₂ - grid discretization in the x₂ direction
    x₁_max - grid size in the x₁ direction
    x₂_max - grid size in the x₂ direction
    source_size - controls size of initial source concentration
    """
    function toy_plume_pde(; wind_str::Float64=1.0, 
                            diffusivity::Float64=1.0, 
                            decay::Float64=0.0,
                            source_loc::Tuple{Float64, Float64}=(5.0, 5.0),
                            Δt::Float64=0.2,
                            t_end::Float64=10.0,
                            Δx₁::Float64=0.5,	
                            Δx₂::Float64=0.5,
                            x₁_max::Float64=10.0,
                            x₂_max::Float64=10.0,
                            source_size::Float64=0.5)
        @parameters  x₁ x₂ t 
        @variables c(..)

        ∂t  = Differential(t)
        ∂x₁  = Differential(x₁)
        ∂x₂  = Differential(x₂)
        ∂²x₁ = Differential(x₁) ^ 2
        ∂²x₂ = Differential(x₂) ^ 2
        U = wind_str
        D = diffusivity
        ℓ₁ = x₁_max
        ℓ₂ = x₂_max
        λ = decay
        c_point = 1.0 #arbitrary, controls initial concentration value

        """
        Hacks the domain of the initial condition to allow for a point source located at the shared mean.
        """
        function gaussian_hack(x₁, 
                                x₂; 
                                μ₁::Float64 = source_loc[1],
                                μ₂::Float64 = source_loc[2], 
                                σ::Float64 = 0.0001)
            return (exp(-((x₁-μ₁)/σ)^2.0/2.0) * exp(-((x₂-μ₂)/σ)^2.0/2.0))
        end


        diff_eq = [∂t(c(x₁, x₂, t)) ~ D*∂²x₁(c(x₁, x₂, t)) + D*∂²x₂(c(x₁, x₂, t))  + U*∂x₁(c(x₁, x₂, t)) - λ*c(x₁, x₂, t)]

        bcs = [
        #initial condition
            c(x₁, x₂, 0) ~ c_point * gaussian_hack(x₁, x₂, σ=source_size),
        #bc's
            ∂x₁(c(ℓ₁, x₂, t)) ~ 0.0,
            ∂x₁(c(0.0, x₂, t)) ~ 0.0,
            ∂x₂(c(x₁, ℓ₂, t)) ~ 0.0,
            ∂x₂(c(x₁, 0.0, t)) ~ 0.0]


        # define space-time plane 
        domains = [x₁ ∈ Interval{:closed, :closed}(0.0, ℓ₁),
                   x₂ ∈ Interval{:closed, :closed}(0.0, ℓ₂), 
                   t ∈ Interval{:closed, :closed}(0.0, t_end), 
                   c(x₁, x₂, t) ∈ Interval{:closed, :closed}(0.0, c_point)]

        # put it all together into a PDE system
        @named pdesys = PDESystem(diff_eq, bcs, domains, [x₁, x₂, t], [c(x₁, x₂, t)]);

        # discretize space [x₁=>Δx₁, x₂=>Δx₂]
        discretization = MOLFiniteDifference([x₁=>Δx₁, x₂=>Δx₂], t)

        # convert the PDE into a system of ODEs via method of lines
        prob = discretize(pdesys, discretization)

        # solve system of ODEs in time.
        sol           = solve(prob, saveat=Δt)
        solution_grid = sol[c(x₁, x₂, t)]

        return solution_grid
end



end