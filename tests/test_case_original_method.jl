using Pkg
# This will be replaced by the module load in the future
Pkg.activate(".")  # Activate the PiCLES package 

using PiCLES
using PiCLES.Operators.core_2D: ParticleDefaults
using PiCLES.Models.WaveGrowthModels2D: WaveGrowth2D
using PiCLES.Simulations
using PiCLES.Grids.CartesianGrid: TwoDCartesianGridMesh, ProjetionKernel, TwoDCartesianGridStatistics

using PiCLES.ParticleSystems: particle_waves_v5 as PW
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

angle1 = pi/4
function u_stopped_angled_line(x, y, t)
  if x==0.
    angle2 = pi/2
  else
    angle2 = atan(y/x)
  end
  dist = sqrt(x^2+y^2)
  opposite = cos(angle1 - angle2) * dist
  adjacent = abs(sin(angle1 - angle2)) * dist
  if opposite <= 125e3
    if adjacent <= 5e3
      return U10 * cos(angle1)
    else
      return 0.0
    end
  else
    return 0.0
  end
end

function v_stopped_angled_line(x, y, t)
  if x==0.
    angle2 = pi/2
  else
    angle2 = atan(y/x)
  end
  dist = sqrt(x^2+y^2)
  opposite = cos(angle1 - angle2) * dist
  adjacent = abs(sin(angle1 - angle2)) * dist
  if opposite <= 125e3
    if adjacent <= 5e3
      return U10 * sin(angle1)
    else
      return 0.0
    end
  else
    return 0.0
  end
end

function u(x, y, t)
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
v(x, y, t) = V10 * 0#(sin(pi*x/50e3))

used_u = u_stopped_angled_line
used_v = v_stopped_angled_line
winds = (u=used_u, v=used_v)

# Define grid
grid = TwoDCartesianGridMesh(600e3, 151, 600e3, 151)
# grid = Grids.SphericalGrid.TwoDSphericalGridMesh(0.0, 180.0, 91, 0, 80.0, 61; periodic_boundary=(true, false))


# Define ODE parameters
ODEpars, Const_ID, Const_Scg = PW.ODEParameters(r_g=r_g0)

# Define particle equations
particle_system = PW.particle_equations(used_u, used_v, γ=Const_ID.γ, q=Const_ID.q);

# Calculate minimal wind sea based on characteristic winds
WindSeamin = FetchRelations.MinimalWindsea(U10, V10, DT)

# Define default particle
default_particle = ParticleDefaults(WindSeamin["lne"], WindSeamin["cg_bar_x"], WindSeamin["cg_bar_y"], 0.0, 0.0)

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
wave_model = WaveGrowth2D(; grid=grid,
    winds=winds,
    ODEsys=particle_system,
    ODEsets=ODE_settings,
    # ODEinit_type=default_particle,
    periodic_boundary=false,
    minimal_particle=FetchRelations.MinimalParticle(U10, V10, DT),
    movie=true)

# Build simulation
wave_simulation = Simulation(wave_model, Δt=DT, stop_time=72hour)#1hours)

# Run simulation
run!(wave_simulation, cash_store=true)

frame_size = (1220, 1080)

max_speeds =  zeros(length(wave_simulation.store.store))
max_energy = maximum([wave_simulation.store.store[i][j,k,1] for i in 1:length(wave_simulation.store.store) for j in 1:wave_model.grid.stats.Nx.N for k in 1:wave_model.grid.stats.Ny.N])
for i in 1:length(wave_simulation.store.store)
  fstate = wave_simulation.store.store[i];
  # plot_particle_collection(fstate, wave_simulation.model.grid)
  # sm2 = wave_model.State[:, :, 3]
  energy = get_tot_energy_domain(fstate)
  p1 = plt.heatmap(grid.data.x[:,1], grid.data.y[1,:], transpose(fstate[:, :, 1]), aspect_ratio=:equal, size=frame_size)
  # p1 = plt.heatmap(p, transpose(sm2), subplot=6, title="State: y momentum ")
  moment_amp= sqrt.(fstate[:,:,2].^2 + fstate[:,:,3].^2)
  c_x = fstate[:,:,2] .* fstate[:,:,1] ./ (2 * moment_amp.^2)
  c_y = fstate[:,:,3] .* fstate[:,:,1] ./ (2 * moment_amp.^2)
  for k in 1:wave_model.grid.stats.Nx.N, l in 1:wave_model.grid.stats.Ny.N
    if isnan(c_x[k,l])
      c_x[k,l] = 0.0
    end
    if isnan(c_y[k,l])
      c_y[k,l] = 0.0
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
                ,clim=(0.0,max_energy*0.5)
  )

  pos_x = wave_simulation.model.grid.stats.xmin + (max_speed_position[1] - 1) * wave_simulation.model.grid.stats.dx
  pos_y = wave_simulation.model.grid.stats.ymin + (max_speed_position[2] - 1) * wave_simulation.model.grid.stats.dy
  plt.scatter!([pos_x], [pos_y], color=:red, markersize=5)

  plt.savefig(p1, "plots/test_case_original/"*string(i)*".png")
  
end

plt.plot(1:length(max_speeds), max_speeds, title="Max speed over time", xlabel="Time step", ylabel="Max speed (m/s)", size=(860, 1080))
plt.savefig("plots/test_case_original/max_speeds_over_time.png")
