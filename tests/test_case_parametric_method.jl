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
U10, V10 = 5.0, 5.0
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
  dist = abs(x - 25e3) # distance from the center line
  if dist <= 5e3
    return U10
  else
    return 0.0
  end
end

function u_stopped_line(x, y, t)
  if y <= 150e3
    dist = abs(x - 25e3) # distance from the center line
    if dist <= 5e3
      return U10
    else
      return 0.0
    end
  else
    return 0.0
  end
end

function u_smoothed_line(x, y, t)
  dist = abs(x - 25e3) # distance from the center line
  if dist <= 5e3
    return U10
  elseif dist <= 10e3
    return U10 * (1 - (dist - 5e3) / 5e3) # linearly decrease to 0 between 5km and 10km
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
u_uniform(x, y, t) = U10
v_uniform(x, y, t) = V10 *0.

used_u = v_uniform
used_v = u_stopped_line
winds = (u=used_u, v=used_v)

# Define grid
grid = TwoDCartesianGridMesh(50e3, 10, 250e3, 10)
# grid = Grids.SphericalGrid.TwoDSphericalGridMesh(0.0, 180.0, 91, 0, 80.0, 61; periodic_boundary=(true, false))


# Define ODE parameters
ODEpars, Const_ID, Const_Scg = PW.ODEParameters(r_g=r_g0)

# Define particle equations
particle_system = PW.particle_equations(used_u, used_v, γ=Const_ID.γ, q=Const_ID.q);

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

max_speeds =  zeros(length(wave_simulation.store.store))
for i in 1:length(wave_simulation.store.store)
  fstate = wave_simulation.store.store[i];
  # plot_particle_collection(fstate, wave_simulation.model.grid)
  # sm2 = wave_model.State[:, :, 3]
  energy = get_tot_energy_domain(fstate)
  p1 = plt.heatmap(grid.data.x[:,1], grid.data.y[1,:], transpose(fstate[:, :, 1]), aspect_ratio=:equal, size=(800, 1920))
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

  max_speed = round(maximum((sqrt.((c_x.*(fstate[:,:,1].>1e-6)).^2 + (c_y.*(fstate[:,:,1].>1e-6)).^2))), digits=4)
  max_speed_position = argmax((sqrt.((c_x.*(fstate[:,:,1].>1e-6)).^2 + (c_y.*(fstate[:,:,1].>1e-6)).^2)))
  max_speeds[i] = max_speed
  plt.plot!(legend=:none,
                title="total energy = "*string(round(energy,digits=3))*"; max_speed = "*string(max_speed)*"; pos = ("*string(max_speed_position[1])*","*string(max_speed_position[2])*")",
                ylabel="y position",
                xlabel="x position",
                xlims=(wave_simulation.model.grid.stats.xmin, wave_simulation.model.grid.stats.xmax),
                ylims=(wave_simulation.model.grid.stats.ymin, wave_simulation.model.grid.stats.ymax)
  )

  pos_x = wave_simulation.model.grid.stats.xmin + (max_speed_position[1] - 1) * wave_simulation.model.grid.stats.dx
  pos_y = wave_simulation.model.grid.stats.ymin + (max_speed_position[2] - 1) * wave_simulation.model.grid.stats.dy
  plt.scatter!([pos_x], [pos_y], color=:red, markersize=5)

  plt.savefig(p1, "plots/test_case_parametric/heatmaps/"*string(i)*".png")
  
end

plt.plot(1:length(max_speeds), max_speeds, title="Max speed over time", xlabel="Time step", ylabel="Max speed (m/s)", size=(860, 1080))
plt.savefig("plots/test_case_parametric/heatmaps/max_speeds_over_time.png")

function fold(v::Vector{Float64})
        return [v[1] v[2] v[4] v[6]; v[2] v[3] v[5] v[7]; v[4] v[5] v[8] v[9]; v[6] v[7] v[9] v[10]]
end

function unfold(M::Matrix{Float64})
        return M[1,1], M[1,2], M[2,2], M[1,3], M[2,3], M[1,4], M[2,4], M[3,3], M[3,4], M[4,4]
end

for i in 1:length(wave_simulation.store.store)
  fstate = wave_simulation.store.store[i]

  
  middle_index = Int(floor(wave_simulation.model.grid.stats.Nx.N / 2))
  cov_sum = zeros(size(fstate)[2])
  cov_cxcx = zeros(size(fstate)[2])
  cov_cxcy = zeros(size(fstate)[2])
  cov_cycy = zeros(size(fstate)[2])
  for j in 1:size(fstate)[2]
    energy = fstate[middle_index, j, 1]
    moment_amp= sqrt.(fstate[middle_index,j,2].^2 + fstate[middle_index,j,3].^2)
    m_c_x = fstate[middle_index,j,2] .* fstate[middle_index,j,1] ./ (2 * moment_amp.^2)
    m_c_y = fstate[middle_index,j,3] .* fstate[middle_index,j,1] ./ (2 * moment_amp.^2)
    m_cov = fold(fstate[middle_index, j, 4:13] .* fstate[middle_index,j,1] ./ (2 * moment_amp.^2))
    m_cov_C = m_cov[1:2, 1:2] *(fstate[middle_index,j,1].>1e-6)
    cov_sum[j] = sum(m_cov_C)
    cov_cxcx[j] = m_cov_C[1,1]
    cov_cxcy[j] = m_cov_C[1,2]
    cov_cycy[j] = m_cov_C[2,2]
  end
  plt.plot(grid.data.y[1,:], cov_sum, title="Sum of covariance matrix elements along y-axis at x = middle",
          label="sum(cov)", legend=:topleft,
          xlabel="y position",
          ylabel="Sum of covariance matrix elements",
          size=(860, 1080),
          ylims=(0,1.75)
  )
  plt.plot!(grid.data.y[1,:], cov_cxcx, label="cov(c_x, c_x)")
  plt.plot!(grid.data.y[1,:], cov_cxcy, label="cov(c_x, c_y)")
  plt.plot!(grid.data.y[1,:], cov_cycy, label="cov(c_y, c_y)")
  plt.savefig("plots/test_case_parametric/covariances/cov_sum_"*string(i)*".png")
end