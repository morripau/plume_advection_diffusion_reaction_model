module SensorPlacementPlots

    using CairoMakie, LinearAlgebra


    """
    plots a sensing grid where:
    S - the set of locations where a sensor can be placed
    U - the set of locations where a sensor cannot be placed but information still gathered
    A - the set of locations where sensors are placed
    """
    function plot_sensor_set(A::Vector, S::Vector, U::Vector; units="position")

        fig = Figure()
        ax = Axis(fig[1,1], aspect=DataAspect(), xlabel="x1 [$(units)]", ylabel="x2 [$(units)]")
    
        scatter!([S[i][1] for i=1:length(S)], [S[i][2] for i=1:length(S)], marker=:x, color="black", label="potential sensor placement")
    
        if length(A) > 0
            scatter!([A[i][1] for i=1:length(A)], [A[i][2] for i=1:length(A)], marker=:circle, color=("white", 0.0), markersize=20, strokewidth=3, strokecolor="red", label="sensor placed")
        end
    
        if length(U) > 0
            scatter!([U[i][1] for i=1:length(U)], [U[i][2] for i=1:length(U)], marker=:star4, color=("white", 0.0), markersize=10, strokewidth=1, strokecolor=("blue", 0.2), label="information only")
        end
    
        fig[1, 2] = Legend(fig, ax, framevisible = false)
    
        return fig
    end

end