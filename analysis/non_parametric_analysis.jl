import Plots as plt
using CSV, DataFrames
import ColorSchemes as cs


function fold(v::Vector{Float64})
        return [v[1] v[2] v[4] v[6]; v[2] v[3] v[5] v[7]; v[4] v[5] v[8] v[9]; v[6] v[7] v[9] v[10]]
end

function unfold(M::Matrix{Float64})
        return M[1,1], M[1,2], M[2,2], M[1,3], M[2,3], M[1,4], M[2,4], M[3,3], M[3,4], M[4,4]
end

function wrap(vector, Nx, Ny)
  res = zeros(Nx, Ny)
  for i in 1:Nx
    for j in 0:(Ny-1)
      res[i,j+1] = vector[i+Nx*j]
    end
  end
  return res
end

function interp(x,y,func,gridX,gridY)
  ix = argmin((-gridX[:] .+ x) .>= 0)-1
  iy = argmin((-gridY[:] .+ y) .>= 0)-1

  dx = gridX[ix+1] - gridX[ix]
  dy = gridY[iy+1] - gridY[iy]
  wx = (x-gridX[ix])/dx
  wy = (y-gridY[iy])/dy

  return wx*wy*func[ix,iy] + (1-wx)*wy*func[ix+1,iy] + wx*(1-wy)*func[ix,iy+1] + (1-wx)*(1-wy)*func[ix+1,iy+1]
end

# READING DATA

println("")
println("---------------- Reading mesh and simulation data ----------------")
println("")
localpath = pwd()
data_path = localpath * "/plots/test_case_parametric"
mesh_sim_data = CSV.read(data_path*"/data/mesh_and_sim_data.csv", DataFrame)

xmin = mesh_sim_data.xmin[1]
xmax = mesh_sim_data.xmax[1]
dx = mesh_sim_data.dx[1]
Nx = mesh_sim_data.Nx[1]
ymin = mesh_sim_data.ymin[1]
ymax = mesh_sim_data.ymax[1]
dy = mesh_sim_data.dy[1]
Ny = mesh_sim_data.Ny[1]
Δt = mesh_sim_data.delta_t[1]
iterations = mesh_sim_data.n_iter[1]

data_x = xmin:dx:xmax
data_y = ymin:dy:ymax
println("     Reading particle data")
particle_data = zeros(iterations, Nx*Ny, 15)
for i in 1:iterations
  particle_data[i,:,:] = Matrix(CSV.read(data_path*"/data/particles/particles_"*string(i)*".csv", DataFrame))[:,2:end]
end
println("         -> done !")
println("")
println("     Reading mesh data")
mesh_data = zeros(iterations, Nx, Ny)
for i in 1:iterations
  mesh_data[i,:,:] = Matrix(CSV.read(data_path*"/data/mesh_values/mesh_values_"*string(i)*".csv", DataFrame))
end
println("         -> done !")
println("")
println("     Reading wind data")
winds = CSV.read(data_path*"/data/wind.csv", DataFrame)
times = winds.t
winds = sqrt.((winds.wind_x .^2) + (winds.wind_y .^2))
println("         -> done !")
println("")


frame_size = (1920,1080)

scale_factor = ones(iterations)
max_speeds =  zeros(iterations)
max_energy = maximum(mesh_data)
max_total_energy = maximum([sum(mesh_data[i,:,:]) for i in 1:iterations])
energies = zeros(iterations)

for i in 1:iterations
  max_speed = sqrt(maximum([particle_data[i,j,2]^2+particle_data[i,j,2]^2 for j in 1:Nx*Ny]))
  max_speeds[i] = max_speed
  fstate = mesh_data[i,:,:]
  energies[i] = sum(fstate)
end


for i in 1:iterations
  if i % 50 == 0
    println(string(i)*"th iteration")
  end
  fstate = mesh_data[i,:,:]
  # plot_particle_collection(fstate, wave_simulation.model.grid)
  # sm2 = wave_model.State[:, :, 3]
  p1 = plt.heatmap(data_x, data_y, transpose(fstate)
                  ,aspect_ratio=:equal
                  ,c=plt.cgrad(:gist_ncar, [0.0,0.03,0.1,0.3,1.0])   #Good candidates : gist_ncar; 
  )
  c_x = wrap(particle_data[i,:,2], Nx, Ny)
  c_y = wrap(particle_data[i,:,3], Nx, Ny)
  M1 = wrap(particle_data[i,:,6], Nx, Ny)
  M2 = wrap(particle_data[i,:,7], Nx, Ny)
  M3 = wrap(particle_data[i,:,8], Nx, Ny)
  M4 = wrap(particle_data[i,:,9], Nx, Ny)
  M5 = wrap(particle_data[i,:,10], Nx, Ny)
  M6 = wrap(particle_data[i,:,11], Nx, Ny)
  M7 = wrap(particle_data[i,:,12], Nx, Ny)
  M8 = wrap(particle_data[i,:,13], Nx, Ny)
  M9 = wrap(particle_data[i,:,14], Nx, Ny)
  M10 = wrap(particle_data[i,:,15], Nx, Ny)
  # for k in 1:Nx, l in 1:Ny
  #   if isnan(c_x[k,l])
  #     c_x[k,l] = 0.0
  #   end
  #   if isnan(c_y[k,l])
  #     c_y[k,l] = 0.0
  #   end
  # end

  max_speed = sqrt(maximum([particle_data[i,j,2]^2+particle_data[i,j,2]^2 for j in 1:Nx*Ny]))
  # c = cgrad([:red,:yellow,:green], [0.50, 0.9995], categorical = false)
  nArrowsX = 12
  nArrowsY = 12
  xlim = (xmin+(xmax-xmin)/(nArrowsX+2), xmax-(xmax-xmin)/(nArrowsX+2))
  ylim = (ymin+(ymax-ymin)/(nArrowsY+2), ymax-(ymax-ymin)/(nArrowsY+2))
  xs = range(xlim...; length=nArrowsX)
  ys = range(ylim...; length=nArrowsY)
  X, Y = reim(complex.(xs', ys))
  Ux = zeros(nArrowsX, nArrowsY)
  Uy = zeros(nArrowsX, nArrowsY)
  for i in 1:(length(xs))
    for j in 1:(length(ys))
        Ux[i,j] = interp(xs[i], ys[j], c_x', data_x, data_y)
        Uy[i,j] = interp(xs[i], ys[j], c_y', data_x, data_y)
        if Ux[i,j] <= 1e-1
          Ux[i,j] = 0
        end
        if Uy[i,j] <= 1e-1
          Uy[i,j] = 0
        end
    end
  end
  scalefactor = (xs[2]-xs[1])/2 /maximum(Ux)
  Ux = scalefactor .* Ux
  Uy = scalefactor .* Uy

  plt.plot!(legend=:none,
                title="Energy distribution in space",
                ylabel="y position",
                xlabel="x position",
                xlims=(xmin, xmax),
                ylims=(ymin, ymax)
                ,clim=(0.0,max_energy*scale_factor[i])
  )
  plt.quiver!(X, Y; quiver=(Ux, Uy), color=:cyan)

  p2 = plt.plot(times[1:(end-1)] ./3600, energies,label=false
                ,xlims=(times[1]/3600,times[end]/3600)
                ,ylims=(0,1.1*max_total_energy)
                ,linewidth=1
                ,ls=:dot
                ,color="#ef8b8b"
  )

  plt.plot!(times[1:i] ./3600, [energies[1:i], energies[1:i]],label=false
                ,ylabel="total energy in the domain"
                ,xlims=(times[1]/3600,times[end]/3600)
                ,ylims=(0,1.1*max_total_energy)
                ,linewidth=[3 3]
                ,ls=[:solid :dot]
                ,color=[:white "#c13030"]
  )

  plt.plot!(plt.twinx(), times[1:(end-1)] ./3600, [winds[1:(end-1)], max_speeds],label=false
                ,title="Wind intensity, wave velocity and energy in time"
                ,ylabel="speed (m/s)"
                ,xlabel="time (hours)"
                ,xlims=(times[1]/3600,times[end]/3600)
                ,ylims=(0,1.1*maximum(winds))
                ,linewidth=[3 3]
                ,ls=:solid
                ,color=["#ffcf9c" "#6ac969"]
  )

  plt.plot!(plt.twinx(), times[1:i] ./3600, [winds[1:i], max_speeds[1:i], NaN.*times[1:i]]
                ,label=["winds speed (m/s)" "wave velocity (m/s)" "energy"]
                ,ylabel="speed (m/s)"
                ,xlims=(times[1]/3600,times[end]/3600)
                ,ylims=(0,1.1*maximum(winds))
                ,linewidth=[5 5 3]
                ,ls=[:solid :solid :dot]
                ,color=["#ff8300" "#31a030" "#c13030"]
  )

  p3 = plt.plot(data_x, [sqrt.(c_x[j,j].^2+c_y[j,j]^2) for j in 1:Nx],legend=false
                # ,size=(frame_size[1],frame_size[2]-200)
                ,title="Wave velocity along the y=x line"
                ,label="wave velocity (m/s)"
                ,ylabel="speed (m/s)"
                ,xlabel="distance (m)"
                ,ylims=(0,5)
                ,linewidth=5
                ,color="blue"
  )

  l1 = @plt.layout [a{0.5h}; b{0.5h}]
  l = @plt.layout [a{0.35w} b{0.7w}]
  p_temp = plt.plot(p2,p3, layout=l1, margin=15plt.mm,top_margin=5plt.mm,
     bottom_margin=5plt.mm)
  p_final = plt.plot(p_temp, p1, layout=l, size=frame_size
                ,margin=15plt.mm
  )

  plt.savefig(p_final, data_path*"/heatmaps/"*string(i)*".png")
end

"""
fstate = wave_simulation.store.store[end]
end_cov_xx = zeros(size(fstate)[2])
end_cov_cxcx = zeros(size(fstate)[2])
end_cov_cxcy = zeros(size(fstate)[2])
end_cov_cycy = zeros(size(fstate)[2])
for i in 1:iterations
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
  plt.plot(data_y[:], cov_sum, title="Covariance matrix elements along y=x-axis",
          label="sum(cov)", legend=:topleft,
          xlabel="y position",
          ylabel="Sum of covariance matrix elements",
          ylims=(0,16),
          size=(860, 1080)
  )
  plt.plot!(data_y[:], cov_cxcx, label="cov(c_x, c_x)")
  plt.plot!(data_y[:], cov_cxcy, label="cov(c_x, c_y)")
  plt.plot!(data_y[:], cov_cycy, label="cov(c_y, c_y)")
  plt.savefig("plots/test_case_parametric/covariances/cov_"*string(i)*".png")

  # plt.plot(data_y[:], cov_cxx, title="Covariance matrix elements along y-axis at x = middle",
  #         label="cov(c_x, x)", legend=:topleft,
  #         xlabel="y position",
  #         ylabel="Sum of covariance matrix elements",
  #         # ylims=(0,7),
  #         size=(860, 1080)
  # )
  # plt.plot!(data_y[:], cov_cxy, label="cov(c_x, y)")
  # plt.plot!(data_y[:], cov_cyx, label="cov(c_y, x)")
  # plt.plot!(data_y[:], cov_cyy, label="cov(c_y, y)")
  # plt.savefig("plots/test_case_parametric/covariances/CX/cov_"*string(i)*".png")

  # plt.plot(data_y[:], cov_xx, title="Covariance matrix elements along y-axis at x = middle",
  #         label="cov(x, x)", legend=:topleft,
  #         xlabel="y position",
  #         ylabel="Sum of covariance matrix elements",
  #         ylims=(0,3e6),
  #         size=(860, 1080)
  # )
  # plt.plot!(data_y[:], cov_xy, label="cov(x, y)")
  # plt.plot!(data_y[:], cov_yy, label="cov(y, y)")
  # plt.savefig("plots/test_case_parametric/covariances/X/cov_"*string(i)*".png")

  if i == iterations
    end_cov_xx = cov_xx
    end_cov_cxcx = cov_cxcx
    end_cov_cxcy = cov_cxcy
    end_cov_cycy = cov_cycy
  end
end

first_swell_index = argmin([end_cov_cxcx[i+1]>end_cov_cxcx[i] for i in 1:(length(end_cov_cxcx)-1)])+1
swell_cov = end_cov_cxcx[first_swell_index:end]
distances = sqrt.(data_y[:] .^2 + data_x[1] .^2)
distances = distances[first_swell_index:end] #.- distances[first_swell_index-5]
plt.plot(distances,(swell_cov), title="Log-log plot of covariance vs y position"
        , xlabel="log(y position)"
        , ylabel="log(covariance)"
        , xaxis=:log, yaxis=:log
        , size=(860, 1080)
)
plt.savefig("plots/test_case_parametric/covariances/0_loglog_cov_C_in_position.png")
"""

println("")
println("Over !")