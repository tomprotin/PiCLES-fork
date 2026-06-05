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
U10, V10 = 10., 10.
DT = 15minutes
r_g0 = 0.85 # ratio of c / c_g (phase velocity/ group velocity).
xmin = 0.
xmax = 6600e3
ymin = 0.0
ymax = 2200e3
Nx = 121
Ny = 41
t_final = 1days

# Define helper functions for particles

function fold(v::Vector{Float64})
        return [v[1] v[2] v[4] v[6]; v[2] v[3] v[5] v[7]; v[4] v[5] v[8] v[9]; v[6] v[7] v[9] v[10]]
end

function unfold(M::Matrix{Float64})
        return M[1,1], M[1,2], M[2,2], M[1,3], M[2,3], M[1,4], M[2,4], M[3,3], M[3,4], M[4,4]
end

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

angle1 = pi/4
function u_stopped_angled_line(x, y, t)
  time_coeff = 0.0
  if t<=10.0*3600
    time_coeff = t / (10.0*3600)
    time_coeff = time_coeff*0.8 + 0.2
  elseif t<=40.0*3600
    time_coeff = 1.0
    time_coeff = time_coeff*0.8 + 0.2
  elseif t <= 50.0*3600
    time_coeff = ((50.0*3600-40.0*3600)-(t-40.0*3600))/(50.0*3600-40.0*3600)
    time_coeff = time_coeff*0.8 + 0.2
  end
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
      return time_coeff * U10 * cos(angle1)
    else
      return 0.0
    end
  else
    return 0.0
  end
end

function v_stopped_angled_line(x, y, t)
  time_coeff = 0.0
  if t<=10*3600
    time_coeff = t / (10*3600)
    time_coeff = time_coeff*0.8 + 0.2
  elseif t<=40*3600
    time_coeff = 1.0
    time_coeff = time_coeff*0.8 + 0.2
  elseif t <= 50*3600
    time_coeff = ((50*3600-40*3600)-(t-40*3600))/(50*3600-40*3600)
    time_coeff = time_coeff*0.8 + 0.2
  end
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
      return time_coeff * U10 * sin(angle1)
    else
      return 0.0
    end
  else
    return 0.0
  end
end


function u_stopped_angled_line_gaussian(x, y, t)
  time_coeff = 0.0
  if t<=50.0*3600
    time_coeff = exp(-0.5*(t-10.0*3600)^2/(15*3600)^2)
  end
  if x==0.
    angle2 = pi/2
  else
    angle2 = atan(y/x)
  end
  dist = sqrt(x^2+y^2)
  opposite = cos(angle1 - angle2) * dist
  adjacent = abs(sin(angle1 - angle2)) * dist
  if opposite <= 125e3
    if adjacent <= 7.5e3
      return time_coeff * U10 * cos(angle1)
    else
      return 0.0
    end
  else
    return 0.0
  end
end


function v_stopped_angled_line_gaussian(x, y, t)
  time_coeff = 0.0
  if t<=50.0*3600
    time_coeff = exp(-0.5*(t-10.0*3600)^2/(15*3600)^2)
  end
  if x==0.
    angle2 = pi/2
  else
    angle2 = atan(y/x)
  end
  dist = sqrt(x^2+y^2)
  opposite = cos(angle1 - angle2) * dist
  adjacent = abs(sin(angle1 - angle2)) * dist
  if opposite <= 125e3
    if adjacent <= 7.5e3
      return time_coeff * U10 * sin(angle1)
    else
      return 0.0
    end
  else
    return 0.0
  end
end


function u_stopped_angled_line_gaussian_rise(x, y, t)
  time_coeff = 1.0
  if t<=24.0*3600
    time_coeff = exp(-0.5*(t-24.0*3600)^2/(12.0*3600)^2)
    time_coeff = 0.2+0.8*time_coeff
  end
  if x==0.
    angle2 = pi/2
  else
    angle2 = atan(y/x)
  end
  dist = sqrt(x^2+y^2)
  opposite = cos(angle1 - angle2) * dist
  adjacent = abs(sin(angle1 - angle2)) * dist
  if opposite <= 125e3
    if adjacent <= 7.5e3
      return time_coeff * U10 * cos(angle1)
    else
      return 0.0
    end
  else
    return 0.0
  end
end


function v_stopped_angled_line_gaussian_rise(x, y, t)
  time_coeff = 1.0
  if t<=24.0*3600
    time_coeff = exp(-0.5*(t-24.0*3600)^2/(12.0*3600)^2)
    time_coeff = 0.2+0.8*time_coeff
  end
  if x==0.
    angle2 = pi/2
  else
    angle2 = atan(y/x)
  end
  dist = sqrt(x^2+y^2)
  opposite = cos(angle1 - angle2) * dist
  adjacent = abs(sin(angle1 - angle2)) * dist
  if opposite <= 125e3
    if adjacent <= 7.5e3
      return time_coeff * U10 * sin(angle1)
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
u_uniform(x, y, t) = U10 *0
v_uniform(x, y, t) = V10 *0.

used_u = u_uniform
used_v = v_uniform
winds = (u=used_u, v=used_v)

# Define grid
grid = TwoDCartesianGridMesh(xmax, Nx, ymax, Ny)
# grid = Grids.SphericalGrid.TwoDSphericalGridMesh(0.0, 180.0, 91, 0, 80.0, 61; periodic_boundary=(true, false))


# Define ODE parameters
ODEpars, Const_ID, Const_Scg = PW.ODEParameters(r_g=r_g0)

# Define particle equations
particle_system = PW.particle_equations(used_u, used_v, γ=Const_ID.γ, q=Const_ID.q,peak_shift=true, dissipation=false);

# Calculate minimal wind sea based on characteristic winds
WindSeamin = FetchRelations.MinimalWindsea(U10, V10, DT)

# Define default particle
initCovarianceMatrix = [1.0 0.0 0.0 0.0; 0.0 1.0 0.0 0.0; 0.0 0.0 grid.stats.dx^2 0.0; 0.0 0.0 0.0 grid.stats.dy^2]
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

using PiCLES.Operators.mapping_2D: reset_PI_u!, ParticleToNode!

fp=0.071
sip=0.015
θ_m = 0/360*2*pi
ncos=4
xm=0.
ym=1100e3
σ_x=500e3
σ_y=500e3
hmax=2.

E_max=(hmax^2)/16
peak_speed = 9.81/(4*pi*fp)
speed_spread = abs(9.81/(4*pi*(sip*0.5-fp))-9.81/(4*pi*(sip*0.5+fp)))

# Build simulation
wave_simulation = Simulation(wave_model, Δt=DT, verbose = true, stop_time=t_final)#1hours)

initialize_simulation!(wave_simulation)

particle_values = (-2, 5.0, 0.3)

function gaussian_space(x, y, mean_x, mean_y, sigma_x, sigma_y)
  return exp(-0.5 * (((x-mean_x)^2)/(sigma_x^2) + ((y-mean_y)^2)/(sigma_y^2)))
end

indices = []
for i in 1:Nx
  for j in 1:Ny
    x = wave_model.ParticleCollection[i,j].position_xy[1]
    y = wave_model.ParticleCollection[i,j].position_xy[2]
    PI = wave_simulation.model.ParticleCollection[i, j]
    energy = E_max * gaussian_space(x, y, xm, ym, σ_x, σ_y)
    current_speed = peak_speed
    cov_matrix = [(speed_spread)^2 0.0 0.0 0.0; 0.0 (current_speed)^2 0.0 0.0; 0.0 0.0 grid.stats.dx^2 0.0; 0.0 0.0 0.0 grid.stats.dy^2]
    ui = [log(energy), current_speed*cos(θ_m), current_speed*sin(θ_m), 0.0, 0.0, unfold(cov_matrix)...]
    reset_PI_u!(PI, ui=ui)
    ParticleToNode!(PI, [0.0,0.0], wave_simulation.model.State, wave_simulation.model.grid, wave_simulation.model.periodic_boundary)
  end
end

# Run simulation
# wave_simulation.model.ParticleCollection.on = true
run!(wave_simulation, cash_store=true)

# ------------ END OF SIMULATION ; BEGIN POST-PROCESSING AND PLOTTING ------------

function interp(x,y,func,gridX,gridY)
  ix = argmin((-gridX[:,1] .+ x) .>= 0)-1
  iy = argmin((-gridY[1,:] .+ y) .>= 0)-1

  dx = gridX[ix+1,1] - gridX[ix,1]
  dy = gridY[1,iy+1] - gridY[1,iy]
  wx = (x-gridX[ix,1])/dx
  wy = (y-gridY[1,iy])/dy

  return wx*wy*func[ix,iy] + (1-wx)*wy*func[ix+1,iy] + wx*(1-wy)*func[ix,iy+1] + (1-wx)*(1-wy)*func[ix+1,iy+1]
end

"""
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
  # c = cgrad([:red,:yellow,:green], [0.50, 0.9995], categorical = false)
  nArrowsX = 25
  nArrowsY = 25
  xlim = (xmin+(xmax-xmin)/(nArrowsX+2), xmax-(xmax-xmin)/(nArrowsX+2))
  ylim = (ymin+(ymax-ymin)/(nArrowsY+2), ymax-(ymax-ymin)/(nArrowsY+2))
  xs = range(xlim...; length=nArrowsX)
  ys = range(ylim...; length=nArrowsY)
  X, Y = reim(complex.(xs', ys))
  Ux = zeros(nArrowsX, nArrowsY)
  Uy = zeros(nArrowsX, nArrowsY)
  for i in 1:(length(xs))
    for j in 1:(length(ys))
        Ux[i,j] = interp(xs[i], ys[j], c_x', grid.data.x, grid.data.y)
        Uy[i,j] = interp(xs[i], ys[j], c_y', grid.data.x, grid.data.y)
    end
  end
  scalefactor = (xs[2]-xs[1])/2 /maximum(Ux)
  Ux = scalefactor .* Ux
  Uy = scalefactor .* Uy

  plt.plot!(legend=:none,
                title="total energy = "*string(round(energy,digits=3))*"; max_speed = "*string(max_speed)*"; pos = ("*string(max_speed_position[1])*","*string(max_speed_position[2])*")",
                ylabel="y position",
                xlabel="x position",
                xlims=(wave_simulation.model.grid.stats.xmin, wave_simulation.model.grid.stats.xmax),
                ylims=(wave_simulation.model.grid.stats.ymin, wave_simulation.model.grid.stats.ymax)
                ,clim=(0.0,max_energy*0.5)
  )
  plt.quiver!(X, Y; quiver=(Ux, Uy), color=:cyan)


  pos_x = wave_simulation.model.grid.stats.xmin + (max_speed_position[1] - 1) * wave_simulation.model.grid.stats.dx
  pos_y = wave_simulation.model.grid.stats.ymin + (max_speed_position[2] - 1) * wave_simulation.model.grid.stats.dy
  plt.scatter!([pos_x], [pos_y], color=:red, markersize=5)

  plt.savefig(p1, "plots/test_case_parametric/heatmaps/"*string(i)*".png")
  
end

plt.plot(1:length(max_speeds), max_speeds, title="Max speed over time", xlabel="Time step", ylabel="Max speed (m/s)", size=(860, 1080))
plt.savefig("plots/test_case_parametric/heatmaps/max_speeds_over_time.png")

fstate = wave_simulation.store.store[end]
end_cov_xx = zeros(size(fstate)[2])
end_cov_cxcx = zeros(size(fstate)[2])
end_cov_cxcy = zeros(size(fstate)[2])
end_cov_cycy = zeros(size(fstate)[2])
for i in 1:length(wave_simulation.store.store)
  fstate = wave_simulation.store.store[i]


  middle_index = Int(floor(wave_simulation.model.grid.stats.Nx.N / 2))
  cov_sum = zeros(size(fstate)[2])
  cov_cxcx = zeros(size(fstate)[2])
  cov_cxcy = zeros(size(fstate)[2])
  cov_cycy = zeros(size(fstate)[2])
  cov_cxx = zeros(size(fstate)[2])
  cov_cxy = zeros(size(fstate)[2])
  cov_cyx = zeros(size(fstate)[2])
  cov_cyy = zeros(size(fstate)[2])
  cov_xx = zeros(size(fstate)[2])
  cov_xy = zeros(size(fstate)[2])
  cov_yy = zeros(size(fstate)[2])
  for j in 1:size(fstate)[2]
    energy = fstate[j, j, 1]
    moment_amp= sqrt.(fstate[j,j,2].^2 + fstate[j,j,3].^2)
    m_c_x = fstate[j,j,2] .* fstate[j,j,1] ./ (2 * moment_amp.^2)
    m_c_y = fstate[j,j,3] .* fstate[j,j,1] ./ (2 * moment_amp.^2)
    m_cov = fold(fstate[j, j, 4:13]) .* fstate[j,j,1] ./ (2 * moment_amp.^2)*(fstate[j,j,1].>1e-8)
    m_cov_C = m_cov[1:2, 1:2] *(fstate[j,j,1].>1e-6)
    cov_sum[j] = sum(m_cov_C)
    cov_cxcx[j] = m_cov[1,1]
    cov_cxcy[j] = m_cov[1,2]
    cov_cycy[j] = m_cov[2,2]
    cov_cxx[j] = m_cov[1,3]
    cov_cxy[j] = m_cov[1,4]
    cov_cyx[j] = m_cov[2,3]
    cov_cyy[j] = m_cov[2,4]
    cov_xx[j] = m_cov[3,3]
    cov_xy[j] = m_cov[3,4]
    cov_yy[j] = m_cov[4,4]
  end
  plt.plot(grid.data.y[1,:], cov_sum, title="Covariance matrix elements along y=x-axis",
          label="sum(cov)", legend=:topleft,
          xlabel="y position",
          ylabel="Sum of covariance matrix elements",
          ylims=(0,16),
          size=(860, 1080)
  )
  plt.plot!(grid.data.y[1,:], cov_cxcx, label="cov(c_x, c_x)")
  plt.plot!(grid.data.y[1,:], cov_cxcy, label="cov(c_x, c_y)")
  plt.plot!(grid.data.y[1,:], cov_cycy, label="cov(c_y, c_y)")
  plt.savefig("plots/test_case_parametric/covariances/cov_"*string(i)*".png")

  # plt.plot(grid.data.y[1,:], cov_cxx, title="Covariance matrix elements along y-axis at x = middle",
  #         label="cov(c_x, x)", legend=:topleft,
  #         xlabel="y position",
  #         ylabel="Sum of covariance matrix elements",
  #         # ylims=(0,7),
  #         size=(860, 1080)
  # )
  # plt.plot!(grid.data.y[1,:], cov_cxy, label="cov(c_x, y)")
  # plt.plot!(grid.data.y[1,:], cov_cyx, label="cov(c_y, x)")
  # plt.plot!(grid.data.y[1,:], cov_cyy, label="cov(c_y, y)")
  # plt.savefig("plots/test_case_parametric/covariances/CX/cov_"*string(i)*".png")

  # plt.plot(grid.data.y[1,:], cov_xx, title="Covariance matrix elements along y-axis at x = middle",
  #         label="cov(x, x)", legend=:topleft,
  #         xlabel="y position",
  #         ylabel="Sum of covariance matrix elements",
  #         ylims=(0,3e6),
  #         size=(860, 1080)
  # )
  # plt.plot!(grid.data.y[1,:], cov_xy, label="cov(x, y)")
  # plt.plot!(grid.data.y[1,:], cov_yy, label="cov(y, y)")
  # plt.savefig("plots/test_case_parametric/covariances/X/cov_"*string(i)*".png")

  if i == length(wave_simulation.store.store)
    end_cov_xx = cov_xx
    end_cov_cxcx = cov_cxcx
    end_cov_cxcy = cov_cxcy
    end_cov_cycy = cov_cycy
  end
end

first_swell_index = argmin([end_cov_cxcx[i+1]>end_cov_cxcx[i] for i in 1:(length(end_cov_cxcx)-1)])+1
swell_cov = end_cov_cxcx[first_swell_index:end]
distances = sqrt.(grid.data.y[1,:] .^2 + grid.data.x[:,1] .^2)
distances = distances[first_swell_index:end] #.- distances[first_swell_index-5]
plt.plot(distances,(swell_cov), title="Log-log plot of covariance vs y position"
        , xlabel="log(y position)"
        , ylabel="log(covariance)"
        , xaxis=:log, yaxis=:log
        , size=(860, 1080)
)
plt.savefig("plots/test_case_parametric/covariances/0_loglog_cov_C_in_position.png")
"""

println("Done !")