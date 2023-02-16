module PlumeModel
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
                            plume_decay::Float64=0.0,
                            source_decay::Float64=0.1,
                            source_loc::Tuple{Float64, Float64}=(5.0, 5.0),
                            Δt::Float64=0.2,
                            t_end::Float64=10.0,
                            Δx₁::Float64=0.5,	
                            Δx₂::Float64=0.5,
                            x₁_max::Float64=10.0,
                            x₂_max::Float64=10.0,
                            source_size::Float64=0.5)
        @parameters  x₁ x₂ t 
        @variables c(..) s(..)

        ∂t  = Differential(t)
        ∂x₁  = Differential(x₁)
        ∂x₂  = Differential(x₂)
        ∂²x₁ = Differential(x₁) ^ 2
        ∂²x₂ = Differential(x₂) ^ 2
        U = wind_str
        D = diffusivity
        ℓ₁ = x₁_max
        ℓ₂ = x₂_max
        λ₁ = plume_decay #plume decay
        λ₂ = source_decay #source decay
        c_point = 1.0 #arbitrary, controls initial concentration value

        """
        Hacks the domain of the initial condition to allow for a point source located at the shared mean.
        """
        function gaussian_hack(x₁, 
                                x₂; 
                                μ₁::Float64 = source_loc[1],
                                μ₂::Float64 = source_loc[2], 
                                σ::Float64 = 0.1)
            return (exp(-((x₁-μ₁)/σ)^2.0) * exp(-((x₂-μ₂)/σ)^2.0))
        end

        function s(x₁, 
            x₂,
            t; 
            μ₁::Float64 = source_loc[1],
            μ₂::Float64 = source_loc[2], 
            σ::Float64 = 0.1)
            #function of t ime and space, continuous source
            #=
            if (x₁ < ℓ₁/2 - source_size/2) && (x₁ > ℓ₁/2 + source_size/2)
                if (x₂ < ℓ₂/2 - source_size/2) && (x₂ > ℓ₂/2 + source_size/2)
                    return c_point*exp(-λ₂*t)
                end
            end
            =#
            return (exp(-((x₁-μ₁)/σ)^2.0) * exp(-((x₂-μ₂)/σ)^2.0))*exp(-λ₂*(t-1))*c_point
        end


        diff_eq = [∂t(c(x₁, x₂, t)) ~ D*∂²x₁(c(x₁, x₂, t)) + D*∂²x₂(c(x₁, x₂, t))  + U*∂x₁(c(x₁, x₂, t)) - λ₁*c(x₁, x₂, t) + s(x₁, x₂, t)]

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

    """
    ****************************************************
    Plume model plots and videos
    ****************************************************
    """

    """
    visualizes a heatmap of gas concentration at a particular time stamp.

    grid - the solution grid output from toy_plume_pde
    time_stamp - the time index
    Δt - time step size
    Δx₁, Δx₂ - grid discretization size
    """
    function viz_heat_map(grid::Array{Float64, 3}; 
                        time_stamp::Int=10, 
                        Δt::Float64=0.2, 
                        Δx₁::Float64=0.5,
                        Δx₂::Float64=0.5,
                        color_range::Tuple{Float64, Float64}=(minimum(grid[:, :, time_stamp]), 
                                                              maximum(grid[:, :, time_stamp])))
        fig = Figure()
        ax = Axis(fig[1, 1], xlabel="x₁ [position]", ylabel="x₂ [position]", xlabelsize = 22, ylabelsize=22, title="time $(truncate(Δt * time_stamp - Δt, 2)) [s]")
        hm = gen_heat_map!(grid, ax, time_stamp=time_stamp, Δt=Δt, color_range=color_range)
        Colorbar(fig[1, 2], hm, label="Fractional Concentration")
        return fig
    end

    function gen_heat_map!(grid::Array{Float64, 3},
                            ax; 
                            time_stamp::Int=10, 
                            Δt::Float64=0.2, 
                            Δx₁::Float64=0.5,
                            Δx₂::Float64=0.5,
                            color_range::Tuple{Float64, Float64}=(minimum(grid[:, :, time_stamp]), 
                                                                  maximum(grid[:, :, time_stamp])))
        x₁_dim = size(grid[:, :, :], 1)
        x₂_dim = size(grid[:, :, :], 2)
        x₁_coords = [Δx₁*i for i=1:x₁_dim]
        x₂_coords = [Δx₂*i for i=1:x₂_dim]
        hm = heatmap!(ax, x₁_coords, x₂_coords, grid[:, :, time_stamp], colorrange=color_range, colormap=:nipy_spectral)
    end

    function truncate(n::Float64, digits::Int)
        n = n*(10^digits)
        n = trunc(Int, n)
        convert(AbstractFloat, n)
        return n/(10^digits)
    end

    """
    generates an mp4 of diffusion titled plume_diffusion.mp4

    grid - the solution grid output from toy_plume_pde
    time_stamp - the time index
    Δt - time step size
    n_frames - should be equal to total_time/Δt
    Δx₁, Δx₂ - grid discretization size
    framerate - frames per second
    """
    function gen_video(grid::Array{Float64, 3}; 
                        Δt::Float64=0.2, 
                        n_frames::Int=50,
                        Δx₁::Float64=0.5,
                        Δx₂::Float64=0.5,
                        framerate::Int=1)
        fig = Figure()
        ax = Axis(fig[1, 1], xlabel="x₁ [position]", ylabel="x₂ [position]", xlabelsize = 30, ylabelsize=30, title="time 0.0 [s]")
        xlims!(1, 10)
        ylims!(1, 10)

        color_range = (minimum(grid[:, :, :]), maximum(grid[:, :, :]))

        t_end = n_frames * Δt
        time_frame_iterator = [i for i=1:length(0.0:Δt:t_end)]

        #, color_range=(minimum(grid[:, :, :]), maximum(grid[:, :, :]))

        record(fig, 
        "plume_diffusion.mp4", 
        time_frame_iterator;
        framerate=framerate) do frame
        time_seconds = Δt * frame - Δt
        ax.title = "time $(truncate(time_seconds, 2)) [s]"
        hm = gen_heat_map!(grid, ax, time_stamp=frame, Δt=Δt, color_range=color_range)
            if frame==1
                Colorbar(fig[1, 2], hm, label="Fractional Concentration")
            end
        end
    end



end