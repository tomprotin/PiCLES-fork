ENV["JULIA_INCREMENTAL_COMPILE"]=true
using Pkg
# This will be replaced by the module load in the future
Pkg.activate(".")  # Activate the PiCLES package 

using PiCLES
using PiCLES.Operators.core_2D_parametric: ParticleDefaultsParam 
using PiCLES.Models.ParametricModels: Parametric2D
using PiCLES.Simulations
using PiCLES.Grids.CartesianGrid: TwoDCartesianGridMesh, ProjetionKernel, TwoDCartesianGridStatistics

using PiCLES.ParticleSystems: particle_waves_v7 as PW
using Oceananigans.Units

# just for simple plotting
import Plots as plt

# Parameters
U10, V10 = 10.0, 10.0
DT = 10minutes
r_g0 = 0.85 # ratio of c / c_g (phase velocity/ group velocity).

# Define wind functions
function ind(x,a,b)
  if x>= a && x<b
    return 1
  else
    return 0
  end
end

function distance(x, y, x0, y0)
  return sqrt((x-x0)^2 + (y-y0)^2)
end

function get_tot_energy_domain(fstate)
        return sum(fstate[:,:,1])
end

function u_line(x, y, t)
  if t <= 300hour
    dist = abs(x - 75e3) # distance from the center of the wind blob
    if dist <= 5e3
      return U10
    else
      return 0.0
    end
  else
    return 0.0
  end
end

function u_sphere(x, y, t)
  if t <= 300hour
    dist = distance(x, y, 75e3, 15e3) # distance from the center of the wind blob
    if dist <= 5e3
      return U10
    else
      return 0.0
    end
  else
    return 0.0
  end
end
v(x, y, t) = V10 * 0.0001#(sin(pi*x/50e3))
winds = (u=v, v=u_line)

# Define grid
grid = TwoDCartesianGridMesh(150e3, 151, 200e3, 201)
# grid = Grids.SphericalGrid.TwoDSphericalGridMesh(0.0, 180.0, 91, 0, 80.0, 61; periodic_boundary=(true, false))


# Define ODE parameters
ODEpars, Const_ID, Const_Scg = PW.ODEParameters(r_g=r_g0)

# Define particle equations
particle_system = PW.particle_equations(u_line, v, γ=Const_ID.γ, q=Const_ID.q);

# Calculate minimal wind sea based on characteristic winds
WindSeamin = FetchRelations.MinimalWindsea(U10, V10, DT)

# Define default particle
initCovarianceMatrix = [1.0 0.0 0.0 0.0; 0.0 1.0 0.0 0.0; 0.0 0.0 1.0 0.0; 0.0 0.0 0.0 1.0]
default_particle = ParticleDefaultsParam(WindSeamin["lne"], WindSeamin["cg_bar_x"], WindSeamin["cg_bar_y"], 0.0, 0.0, initCovarianceMatrix)

# Define ODE settings
ODE_settings = PW.ODESettings(
  Parameters=ODEpars,
  # define mininum energy threshold
  log_energy_minimum=WindSeamin["lne"],
  saving_step=DT,
  timestep=DT,
  total_time=T = 6days,
  dt=1e-3, 
  dtmin=1e-4, 
  force_dtmin=true)

# Build wave model
wave_model = Parametric2D(; grid=grid,
    winds=winds,
    ODEsys=particle_system,
    ODEsets=ODE_settings,
    # ODEinit_type=default_particle,
    periodic_boundary=false,
    minimal_particle=FetchRelations.MinimalParticle(U10, V10, DT),
    movie=true)

# Build simulation
wave_simulation = Simulation(wave_model, Δt=DT, verbose = true, stop_time=24hour)#1hours)

# Run simulation
run!(wave_simulation, cash_store=true)

# Plot final state
fstate = wave_simulation.store.store[end];
p1 = plt.heatmap(grid.data.x[:,1] / 1e3, grid.data.y[1,:] / 1e3, fstate[:, :, 1])

# function plot_particle_collection(state_i, grid)
#     # particles = wave_model.ParticleCollection
#     p = plt.plot(layout=(3, 2), size=(1200, 1000))
#     # heatmap!(p, transpose(particles.on), subplot=1, title="on | iter=" * string(wave_model.clock.iteration) * " | time=" * string(wave_model.clock.time))
#     # heatmap!(p, transpose(particles.boundary), subplot=2, title="boundary")

#     sE = state_i[:, :, 1]
#     sE[grid.data.mask.==0] .= NaN
#     sE[grid.data.mask.==2] .= NaN
#     plt.heatmap!(p, transpose(sE), subplot=3, title="State: Energy", clims=(0, NaN))

#     sm1 = state_i[:, :, 2]
#     sm1[grid.data.mask.==0] .= NaN
#     sm1[grid.data.mask.==2] .= NaN
#     plt.heatmap!(p, transpose(sm1), subplot=4, title="State: x momentum ", clims=(0, NaN))

#     sm2 = state_i[:, :, 3]
#     sm2[grid.data.mask.==0] .= NaN
#     sm2[grid.data.mask.==2] .= NaN
#     plt.heatmap!(p, transpose(sm2), subplot=6, title="State: y momentum ")
#     # title = plot!(title="Plot title", grid=false, showaxis=false, bottom_margin=-50Plots.px)
#     display(p)
#     return p
# end


  # fstate = wave_simulation.store.store[end];
  # plot_particle_collection(fstate, wave_simulation.model.grid)


for i in 1:length(wave_simulation.store.store)
  fstate = wave_simulation.store.store[i];
  # plot_particle_collection(fstate, wave_simulation.model.grid)
  # sm2 = wave_model.State[:, :, 3]
  energy = get_tot_energy_domain(fstate)
  p1 = plt.heatmap(grid.data.x[:,1], grid.data.y[1,:], transpose(fstate[:, :, 1]), aspect_ratio=:equal, size=(860, 1080))
  # p1 = plt.heatmap(p, transpose(sm2), subplot=6, title="State: y momentum ")
  moment_amp= sqrt.(fstate[:,:,2].^2 + fstate[:,:,3].^2)
  c_x = fstate[:,:,2] .* fstate[:,:,1] ./ (2 * moment_amp.^2)
  c_y = fstate[:,:,3] .* fstate[:,:,1] ./ (2 * moment_amp.^2)
  for i in 1:wave_model.grid.stats.Nx.N, j in 1:wave_model.grid.stats.Ny.N
    if isnan(c_x[i,j])
      c_x[i,j] = 0.0
    end
    if isnan(c_y[i,j])
      c_y[i,j] = 0.0
    end
  end
  max_speed = round(maximum(sqrt.(c_x.^2 + c_y.^2)), digits=4)
  max_speed_position = argmax(sqrt.(c_x.^2 + c_y.^2))

  plt.plot!(legend=:none,
                title="total energy = "*string(round(energy,digits=3))*"; max_speed = "*string(max_speed)*"; pos = ("*string(max_speed_position[1])*","*string(max_speed_position[2])*")",
                ylabel="y position",
                xlabel="x position",
                xlims=(wave_simulation.model.grid.stats.xmin, wave_simulation.model.grid.stats.xmax),
                ylims=(wave_simulation.model.grid.stats.ymin, wave_simulation.model.grid.stats.ymax))

  plt.savefig(p1, "plots/test_case_parametric/"*string(i)*".png")
  
end