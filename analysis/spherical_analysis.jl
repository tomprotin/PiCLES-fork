import Plots as plt
using CSV, DataFrames
import ColorSchemes as cs
using FFMPEG_jll
using LinearAlgebra
using GeoDatasets
using Colors
# using CairoMakie

# ============================================================================
# USER PARAMETERS — edit here, nothing below needs to change
# ============================================================================

# --- Simulation output directory ---
# data_path = pwd() * "/plots/good_test_cases/test_case_parametric"
data_path = pwd() * "/plots/test_case_parametric"

# --- Globe camera (fixed views) ---
cam_az = 95    # azimuth (°): default view centres ~lon 100°E
cam_el = 18    # elevation (°): tilt above the equatorial plane

# --- Globe display mode ---
# false → two fixed half-globes (front + back hemisphere)
# true  → single globe tracking the peak-energy cell every frame
focus_globe = true

# --- Probe locations for 2D wave spectra ---
# The last probe is dynamic: its position is overridden every frame to follow
# the peak ocean energy cell (its initial value here is ignored).
probe_lons   = [45.0, 115.0, 165.0, -145.0]   # °E
probe_lats   = [ -30.0,  -50.0,  -35.0,   10.0]   # °N  (last entry ignored)
probe_colors = [:cyan, :lime, :magenta, :coral]

# --- Wave-speed arrows ---
N_skip_arr     = 6      # sample every N grid cells (lower = denser arrows)
arrow_scale_ws = 0.1    # fixed screen-space length (sphere radius = 1)

# --- 2D spectrum panels ---
N_spec     = 80
spec_cgrad = plt.cgrad(:plasma)

# --- Thresholds (as fraction of global max) ---
E_threshold_ws_frac = 0.0001   # cells below max_energy * frac are masked
spec_c_max_frac     = 1.4      # spectrum velocity axis = max_ws_global * frac

# --- Histogram ---
n_hist_bins = 15

# --- Focus globe smoothing ---
# Gaussian kernel half-width in frames. Higher = smoother panning, more lag.
focus_smooth_σ = 8.0

# --- Movie frame rate ---
framerate = 60   # frames per second for output MP4 files

# ============================================================================
# spherical_analysis.jl
#
# Analysis and visualisation script for the parametric PiCLES test case
# run on a spherical (lon/lat) grid.
#
# Adapted from non_parametric_analysis.jl (Cartesian version).
#
# Key differences from the Cartesian version:
#   - Grid axes are longitude (°E) and latitude (°N) instead of x/y in metres.
#   - Heatmaps use the :geo projection via the GeoPlots / Proj backend so that
#     the lat/lon grid is shown on a proper map with coastlines.
#   - Quiver arrows are scaled to degrees of arc rather than metres.
#   - The covariance diagnostic traverses a great-circle band eastward from the
#     storm centre instead of the Cartesian y=x diagonal.
#   - All spatial labels are in degrees.
# ============================================================================


# ----------------------------------------------------------------------------
# Helper functions (identical to Cartesian version)
# ----------------------------------------------------------------------------

"""
    fold(v) → 4×4 symmetric matrix

Reconstruct the symmetric covariance matrix from its 10 upper-triangle entries.
Ordering: (1,1),(1,2),(2,2),(1,3),(2,3),(1,4),(2,4),(3,3),(3,4),(4,4)
"""
function fold(v::Vector{Float64})
    return [v[1] v[2] v[4] v[6];
            v[2] v[3] v[5] v[7];
            v[4] v[5] v[8] v[9];
            v[6] v[7] v[9] v[10]]
end

"""
    unfold(M) → 10-element vector (upper triangle of 4×4 symmetric M)
"""
function unfold(M::Matrix{Float64})
    return M[1,1], M[1,2], M[2,2], M[1,3], M[2,3],
           M[1,4], M[2,4], M[3,3], M[3,4], M[4,4]
end

"""
    wrap(vector, Nx, Ny) → Nx×Ny matrix

Reshape a flat particle-data vector (stored column-major, x-first) into a
2D grid array.
"""
function wrap(vector, Nx, Ny)
    res = zeros(Nx, Ny)
    for i in 1:Nx
        for j in 0:(Ny-1)
            res[i, j+1] = vector[i + Nx*j]
        end
    end
    return res
end

"""
    interp_bilinear(lon, lat, field, grid_lons, grid_lats) → Float64

Bilinear interpolation of `field` (Nlon×Nlat) at (lon, lat).
`grid_lons` and `grid_lats` are 1D sorted arrays of grid coordinates in degrees.
Handles longitude periodicity by wrapping around 360°.
"""
function interp_bilinear(lon, lat, field, grid_lons, grid_lats)
    # Wrap lon into [grid_lons[1], grid_lons[end]]
    span = grid_lons[end] - grid_lons[1]
    lon  = grid_lons[1] + mod(lon - grid_lons[1], span)

    # Find bounding indices
    ilon = searchsortedlast(grid_lons, lon)
    ilat = searchsortedlast(grid_lats, lat)

    ilon = clamp(ilon, 1, length(grid_lons) - 1)
    ilat = clamp(ilat, 1, length(grid_lats) - 1)

    dlon = grid_lons[ilon+1] - grid_lons[ilon]
    dlat = grid_lats[ilat+1] - grid_lats[ilat]
    wlon = (lon - grid_lons[ilon]) / dlon
    wlat = (lat - grid_lats[ilat]) / dlat

    return (      wlon *       wlat  * field[ilon+1, ilat+1]
            + (1-wlon) *       wlat  * field[ilon,   ilat+1]
            +      wlon * (1-wlat)   * field[ilon+1, ilat  ]
            + (1-wlon) * (1-wlat)    * field[ilon,   ilat  ])
end


"""
    ortho_image(energy_field, data_lon, data_lat, cam_az_deg, cam_el_deg; N=300)

Raytrace an orthographic projection of the visible hemisphere onto an N×N image.
Returns (img, s) where img[i,j] is the interpolated energy at screen pixel (s[i], s[j])
and s ∈ [-1, 1].  Pixels outside the sphere disk are left at 0.

This sidesteps the GR backend limitation that parametric surface() plots do not
support custom per-vertex colours.
"""
function ortho_image(energy_field, data_lon, data_lat, cam_az_deg, cam_el_deg; N=300)
    az = cam_az_deg * π / 180
    el = cam_el_deg * π / 180
    # Eye direction (unit vector toward viewer)
    ex, ey, ez = cos(el)*cos(az), cos(el)*sin(az), sin(el)
    # Screen-right and screen-up basis vectors (orthonormal to eye)
    rx, ry, rz = -sin(az),          cos(az),         0.0
    ux, uy, uz = -sin(el)*cos(az), -sin(el)*sin(az),  cos(el)

    s   = range(-1.0, 1.0, length=N)
    img = fill(NaN, N, N)   # NaN outside disk → transparent background
    for (i, xp) in enumerate(s)
        for (j, yp) in enumerate(s)
            xp^2 + yp^2 >= 1.0 && continue   # outside sphere disk
            d  = sqrt(1 - xp^2 - yp^2)
            # 3-D point on the visible hemisphere surface
            px = xp*rx + yp*ux + d*ex
            py = xp*ry + yp*uy + d*ey
            pz = xp*rz + yp*uz + d*ez
            lat = asin(clamp(pz, -1.0, 1.0)) * 180 / π
            lon = atan(py, px)              * 180 / π
            img[i, j] = interp_bilinear(lon, lat, energy_field, data_lon, data_lat)
        end
    end
    return img, collect(s)
end


"""
    project_parallel(φ_deg, cam_az_deg, cam_el_deg; n=300)

Project a constant-latitude circle onto orthographic screen coordinates.
Returns (xs, ys); NaN marks horizon breaks where the arc crosses to the back hemisphere.
"""
function project_parallel(φ_deg, cam_az_deg, cam_el_deg; n=300)
    az, el = cam_az_deg * π/180, cam_el_deg * π/180
    ex, ey, ez = cos(el)*cos(az), cos(el)*sin(az), sin(el)
    rx, ry, rz = -sin(az),  cos(az),  0.0
    ux, uy, uz = -sin(el)*cos(az), -sin(el)*sin(az), cos(el)
    φ = φ_deg * π/180
    xs, ys = Float64[], Float64[]
    for lon in range(0.0, 2π*(1 - 1/n), length=n)
        px, py, pz = cos(φ)*cos(lon), cos(φ)*sin(lon), sin(φ)
        if px*ex + py*ey + pz*ez >= 0
            push!(xs, px*rx + py*ry + pz*rz)
            push!(ys, px*ux + py*uy + pz*uz)
        else
            push!(xs, NaN); push!(ys, NaN)
        end
    end
    return xs, ys
end


"""
    project_meridian(λ_deg, cam_az_deg, cam_el_deg; n=300)

Project a constant-longitude great-circle arc (meridian) onto orthographic screen
coordinates. Returns (xs, ys); NaN marks breaks at the horizon.
"""
function project_meridian(λ_deg, cam_az_deg, cam_el_deg; n=300)
    az, el = cam_az_deg * π/180, cam_el_deg * π/180
    ex, ey, ez = cos(el)*cos(az), cos(el)*sin(az), sin(el)
    rx, ry, rz = -sin(az),  cos(az),  0.0
    ux, uy, uz = -sin(el)*cos(az), -sin(el)*sin(az), cos(el)
    λ = λ_deg * π/180
    xs, ys = Float64[], Float64[]
    for lat in range(-π/2, π/2, length=n)
        px, py, pz = cos(lat)*cos(λ), cos(lat)*sin(λ), sin(lat)
        if px*ex + py*ey + pz*ez >= 0
            push!(xs, px*rx + py*ry + pz*rz)
            push!(ys, px*ux + py*uy + pz*uz)
        else
            push!(xs, NaN); push!(ys, NaN)
        end
    end
    return xs, ys
end


"""
    project_axis(cam_el_deg; extend=1.5, pierce=0.25)

Returns two segments that together create the "globe on a stick" illusion:
  - seg_north: starts `pierce` screen-units below the north-pole projection
    and extends to the top tip. The short inside-disk part renders on top of
    the heatmap near the pole, making the stick visibly pierce through.
  - seg_south: the bottom tip below the sphere disk (south pole is on the back
    hemisphere, so only the outside segment is drawn).
"""
function project_axis(cam_el_deg; extend=1.5, pierce=0.0)
    uz  = cos(cam_el_deg * π/180)
    top = extend * uz
    seg_north = ([0.0, 0.0], [uz - pierce, top])   # pole vicinity + top tip
    seg_south = ([0.0, 0.0], [-1.0,        -top])  # bottom tip only
    return seg_north, seg_south
end


"""
    project_arrows(cx_field, cy_field, energy_field, data_lon, data_lat,
                   i_arr, j_arr, cam_az_deg, cam_el_deg; E_threshold, arrow_scale)

Sample a sparse lon/lat grid and project the group-velocity direction (cx, cy) onto
orthographic screen coordinates. Only includes points that are:
  - on the visible hemisphere (front face)
  - not too close to the disk limb (avoids foreshortening artefacts)
  - above `E_threshold` in energy

Returns (xs, ys, us, vs) ready for plt.quiver!(xs, ys, quiver=(us, vs)).
Arrows have fixed screen-space length `arrow_scale` (direction only, not magnitude).

Note: cx_field / cy_field can be any field proportional to the velocity direction
(e.g. raw momentum components m_x, m_y share the same direction as c̄_x, c̄_y).
"""
function project_arrows(cx_field, cy_field, energy_field, data_lon, data_lat,
                        i_arr, j_arr, cam_az_deg, cam_el_deg;
                        E_threshold::Float64, arrow_scale::Float64=0.05)
    az = cam_az_deg * π/180;  el = cam_el_deg * π/180
    ex, ey, ez = cos(el)*cos(az), cos(el)*sin(az), sin(el)
    rx, ry, rz = -sin(az),  cos(az),  0.0
    ux, uy, uz = -sin(el)*cos(az), -sin(el)*sin(az), cos(el)

    xs, ys, us, vs = Float64[], Float64[], Float64[], Float64[]

    for ii in i_arr, jj in j_arr
        energy_field[ii, jj] < E_threshold && continue

        lon = data_lon[ii] * π/180
        lat = data_lat[jj] * π/180

        px = cos(lat)*cos(lon);  py = cos(lat)*sin(lon);  pz = sin(lat)

        # Skip back hemisphere
        px*ex + py*ey + pz*ez < 0.0 && continue

        sx = px*rx + py*ry + pz*rz
        sy = px*ux + py*uy + pz*uz

        # Skip near the limb to avoid foreshortening artefacts
        sx^2 + sy^2 > 0.92^2 && continue

        cx = cx_field[ii, jj];  cy = cy_field[ii, jj]
        spd = sqrt(cx^2 + cy^2)
        spd < 1e-6 && continue

        # Eastward and northward tangent unit vectors at this lon/lat
        elon_x, elon_y, elon_z =  -sin(lon),           cos(lon),          0.0
        elat_x, elat_y, elat_z = (-sin(lat)*cos(lon)), (-sin(lat)*sin(lon)), cos(lat)

        # 3D unit velocity direction on the sphere
        vx = cx/spd * elon_x + cy/spd * elat_x
        vy = cx/spd * elon_y + cy/spd * elat_y
        vz = cx/spd * elon_z + cy/spd * elat_z

        # Project onto screen plane and normalise (remove foreshortening)
        dsx = vx*rx + vy*ry + vz*rz
        dsy = vx*ux + vy*uy + vz*uz
        dlen = sqrt(dsx^2 + dsy^2)
        dlen < 1e-6 && continue

        push!(xs, sx);              push!(ys, sy)
        push!(us, dsx/dlen * arrow_scale)
        push!(vs, dsy/dlen * arrow_scale)
    end

    return xs, ys, us, vs
end


"""
    project_point_to_screen(lon_deg, lat_deg, cam_az_deg, cam_el_deg)

Project a single lon/lat point onto orthographic screen coordinates.
Returns `(sx, sy)` if on the visible hemisphere, `nothing` if behind the globe.
"""
function project_point_to_screen(lon_deg, lat_deg, cam_az_deg, cam_el_deg)
    az, el = cam_az_deg * π/180, cam_el_deg * π/180
    ex, ey, ez = cos(el)*cos(az), cos(el)*sin(az), sin(el)
    rx, ry, rz = -sin(az), cos(az), 0.0
    ux, uy, uz = -sin(el)*cos(az), -sin(el)*sin(az), cos(el)
    lon, lat   = lon_deg * π/180, lat_deg * π/180
    px, py, pz = cos(lat)*cos(lon), cos(lat)*sin(lon), sin(lat)
    px*ex + py*ey + pz*ez < 0 && return nothing
    return (px*rx + py*ry + pz*rz,  px*ux + py*uy + pz*uz)
end


"""
    create_movie(frame_dir, output_path; framerate=12, pattern="%d.png")

Stitch sequentially-numbered PNG frames in `frame_dir` into an MP4 using FFmpeg.
Frames must be named 1.png, 2.png, … (matching `pattern`).
Requires `ffmpeg` to be available on PATH.
"""
function create_movie(frame_dir::String, output_path::String;
                      framerate::Int = 12,
                      pattern::String = "%d.png")
    input        = joinpath(frame_dir, pattern)
    scale_filter = "scale=trunc(iw/2)*2:trunc(ih/2)*2"   # ensure even dimensions for libx264
    run(`$(FFMPEG_jll.ffmpeg()) -y -framerate $framerate -start_number 1
                -i $input
                -vf $scale_filter
                -c:v libx264 -pix_fmt yuv420p -crf 18
                $output_path`)
    println("Movie saved: $output_path")
end


# ============================================================================
# 1. Read simulation metadata and data
# ============================================================================

println("")
println("---------- Reading mesh and simulation data (spherical) ----------")
println("")

mesh_sim_data = CSV.read(data_path * "/data/mesh_and_sim_data.csv", DataFrame)

# Grid bounds and resolution (degrees)
lon_min = mesh_sim_data.xmin[1]
lon_max = mesh_sim_data.xmax[1]
dlon    = mesh_sim_data.dx[1]       # degrees
Nlon    = mesh_sim_data.Nx[1]

lat_min = mesh_sim_data.ymin[1]
lat_max = mesh_sim_data.ymax[1]
dlat    = mesh_sim_data.dy[1]       # degrees
Nlat    = mesh_sim_data.Ny[1]

Δt         = mesh_sim_data.delta_t[1]
iterations = mesh_sim_data.n_iter[1]

# 1D coordinate arrays (degrees)
data_lon = collect(range(lon_min, lon_max, length=Nlon))
data_lat = collect(range(lat_min, lat_max, length=Nlat))

println("   Grid: $(Nlon) × $(Nlat), lon [$(lon_min)°, $(lon_max)°], lat [$(lat_min)°, $(lat_max)°]")
println("   Resolution: Δlon=$(round(dlon,digits=2))°, Δlat=$(round(dlat,digits=2))°")
println("")

# ---- Particle data  (iterations × Nlon*Nlat × 15) -------------------------
println("   Reading particle data …")
let _df0 = CSV.read(data_path * "/data/particles/particles_1.csv", DataFrame)
    n_csv = nrow(_df0)
    n_expected = Nlon * Nlat
    if n_csv != n_expected
        error("Particle CSV has $n_csv rows but grid is $(Nlon)×$(Nlat)=$n_expected. " *
              "Delete old CSV files and re-run the simulation.")
    end
end
particle_data = zeros(iterations, Nlon * Nlat, 15)
for i in 1:iterations
    particle_data[i, :, :] = Matrix(
        CSV.read(data_path * "/data/particles/particles_" * string(i) * ".csv",
                 DataFrame))[:, 2:end]
end
println("      → done !")
println("")

# ---- Mesh energy data  (iterations × Nlon × Nlat) -------------------------
println("   Reading mesh data …")
mesh_data = zeros(iterations, Nlon, Nlat)
for i in 1:iterations
    mesh_data[i, :, :] = (Matrix(
        CSV.read(data_path * "/data/mesh_values/mesh_values_" * string(i) * ".csv",
                 DataFrame)))'
end
println("      → done !")
println("")

# ---- Wind time series -------------------------------------------------------
println("   Reading wind data …")
winds_df   = CSV.read(data_path * "/data/wind.csv", DataFrame)
times      = winds_df.t                                         # seconds
println("      → done !")
println("")


# ============================================================================
# 2. Pre-compute per-iteration scalar diagnostics
# ============================================================================

frame_size     = (1920, 1080)
max_energy     = maximum(mesh_data)
max_total_energy = maximum([sum(mesh_data[i, :, :]) for i in 1:iterations])
total_energies = [sum(mesh_data[i, :, :]) for i in 1:iterations]
max_speeds     = [sqrt(maximum(particle_data[i, :, 2] .^ 2 .+
                               particle_data[i, :, 3] .^ 2))
                  for i in 1:iterations]


# ============================================================================
# 3. Frame-by-frame plots
#    Layout mirrors the Cartesian version:
#      left panel  (35 %) : time series — total energy, wind speed, max wave speed
#      right panel (65 %) : geographic map — energy heatmap + group velocity quiver
# ============================================================================

mkpath(data_path * "/heatmaps")
mkpath(data_path * "/covariances")

# Camera: look at the swell source region (lon≈100°, lat≈10°) from slightly above.
# Azimuth rotates the view around the Z-axis; elevation tilts up from the equatorial plane.
cam_az2 = cam_az + 180   # antipodal camera (back hemisphere, derived from cam_az)

# Sphere border circle (shared by both globes)
θ_c = range(0, 2π, length=300)

# Pre-compute graticule lines (meridians every 30°) and equator
lon_lines  = collect(0:30:330)
graticule1 = [project_meridian(λ, cam_az,  cam_el) for λ in lon_lines]
equator1   = project_parallel(0.0, cam_az,  cam_el)
axis1_north, axis1_south = project_axis(cam_el)

graticule2 = [project_meridian(λ, cam_az2, cam_el) for λ in lon_lines]
equator2   = project_parallel(0.0, cam_az2, cam_el)
axis2_north, axis2_south = project_axis(cam_el)   # same elevation → same segments

# Non-linear colormap: rapid colour transitions at low energy (vivid even when
# swell has spread and peak is well below max_energy).
custom_cgrad = plt.cgrad(:gist_ncar, [0.0, 0.02, 0.08, 0.20, 0.50, 1.0])

# Standalone colorbar strip (static, computed once outside the loop).
E_axis = collect(range(0.0, max_energy, length=256))
p_cb = plt.heatmap(
    [0.0], E_axis, reshape(E_axis, 1, :),
    color        = custom_cgrad,
    clims        = (0.0, max_energy),
    colorbar     = false,
    xticks       = false,
    yaxis        = :right,
    ylabel       = "Energy [m²]",
    framestyle   = :box,
    aspect_ratio = :auto,
)

# ============================================================================
# Land mask: raytrace onto both globe screens (done once — camera is fixed).
# Land pixels are baked into each frame's image using a sentinel value, which
# the extended colormaps map to a solid land colour.
# ============================================================================

lsm_lon_gd, lsm_lat_gd, lsm_data_gd = GeoDatasets.landseamask(resolution='c', grid=5)

# land_field[i,j] = 1.0 where land, 0.0 where ocean  (Float for ortho_image)
land_field = [begin
    lo_w = mod(data_lon[i] + 180.0, 360.0) - 180.0
    ii   = argmin(abs.(lsm_lon_gd .- lo_w))
    jj   = argmin(abs.(lsm_lat_gd .- data_lat[j]))
    Float64(lsm_data_gd[ii, jj])
end for i in 1:Nlon, j in 1:Nlat]

# Raytrace land mask to screen space at the same resolution as the energy images
land_img1, _ = ortho_image(land_field, data_lon, data_lat, cam_az,  cam_el; N=300)
land_img2, _ = ortho_image(land_field, data_lon, data_lat, cam_az2, cam_el; N=300)

# Precomputed boolean masks (land pixels inside the sphere disk)
land_visible1 = (.!isnan.(land_img1)) .& (land_img1 .> 0.5)
land_visible2 = (.!isnan.(land_img2)) .& (land_img2 .> 0.5)
println("   Land: $(sum(land_visible1)) px on front globe, $(sum(land_visible2)) px on back globe.")

# --- Extended colormaps -------------------------------------------------------
# Strategy: extend clims below the true minimum with a "land sentinel" value.
# The colormap maps the sentinel range to solid land colour and the rest to the
# existing ocean colormap.  Fraction of colormap for land = land_frac = 0.5.

# Energy sentinel and colormap (wave-speed equivalents deferred to section 4
# where min_ws / max_ws_global are first available).
LAND_SENTINEL_E  = -max_energy   # energy: range [-E, E] → land maps to the lower half

# Build the energy colormap by prepending a land colour to the ocean cgrad.
_ref_col   = custom_cgrad[0.5]
CGRAD_RGBA = typeof(_ref_col)
saddlebrown = CGRAD_RGBA(139/255, 69/255, 19/255, 1.0)

n_c    = 200
c_pts  = collect(range(0.0, 1.0, length=n_c))
# Positions in the extended [0,1] space: land at [0, 0.5), ocean at [0.5, 1.0]
land_pts  = [0.0, 0.5 - 1e-5]
ocean_pts = [0.5 + p * 0.5 for p in c_pts]
full_pts  = vcat(land_pts, ocean_pts)

e_ocean_cols  = [custom_cgrad[p] for p in c_pts]
land_energy_cgrad = plt.cgrad(vcat(fill(saddlebrown, 2), e_ocean_cols), full_pts)


# -------------------------------------------------------------------------
# Pre-compute smoothed focus-globe camera path (used when focus_globe=true).
# Raw path = argmax(energy) per frame; smoothed with a Gaussian kernel to
# avoid the jerky discrete jumps between grid cells.
# -------------------------------------------------------------------------
function _smooth_vec(v::Vector{Float64}, σ::Float64)
    out = similar(v)
    hw  = ceil(Int, 3σ)
    for i in eachindex(v)
        wsum = 0.0; vsum = 0.0
        for j in max(firstindex(v), i - hw):min(lastindex(v), i + hw)
            w = exp(-0.5 * ((j - i) / σ)^2)
            wsum += w; vsum += w * v[j]
        end
        out[i] = vsum / wsum
    end
    return out
end

let az_raw = zeros(iterations), el_raw = zeros(iterations)
    for i in 1:iterations
        ci = argmax(mesh_data[i, :, :])
        az_raw[i] = data_lon[ci[1]]
        el_raw[i] = clamp(Float64(data_lat[ci[2]]), -75.0, 75.0)
    end
    # Smooth elevation directly (not circular)
    global cam_el_focus = _smooth_vec(el_raw, focus_smooth_σ)
    # Smooth azimuth via unit complex numbers to handle the 0°/360° wrap
    az_z  = exp.(im .* az_raw .* (π / 180))
    az_re = _smooth_vec(real.(az_z), focus_smooth_σ)
    az_im = _smooth_vec(imag.(az_z), focus_smooth_σ)
    global cam_az_focus = mod.(angle.(az_re .+ im .* az_im) .* (180 / π), 360.0)
end


for i in 1:iterations

    if i % 10 == 0
        println("Frame " * string(i) * " / " * string(iterations))
    end

    fstate = mesh_data[i, :, :]   # (Nlon × Nlat) energy field

    # ------------------------------------------------------------------
    # Globe panel(s): controlled by `focus_globe`
    # ------------------------------------------------------------------

    if focus_globe
        # Single globe: use pre-smoothed camera path
        cam_az_dyn = cam_az_focus[i]
        cam_el_dyn = cam_el_focus[i]

        grat_dyn  = [project_meridian(λ, cam_az_dyn, cam_el_dyn) for λ in lon_lines]
        eq_dyn    = project_parallel(0.0, cam_az_dyn, cam_el_dyn)

        lnd_dyn, _   = ortho_image(land_field, data_lon, data_lat, cam_az_dyn, cam_el_dyn; N=300)
        lvis_dyn     = (.!isnan.(lnd_dyn)) .& (lnd_dyn .> 0.5)
        img_dyn, s_px = ortho_image(fstate, data_lon, data_lat, cam_az_dyn, cam_el_dyn; N=300)
        img_dyn[lvis_dyn] .= LAND_SENTINEL_E

        p_globe = plt.heatmap(
            s_px, s_px, img_dyn',
            color        = land_energy_cgrad,
            clims        = (LAND_SENTINEL_E, max_energy),
            aspect_ratio = :equal,
            title        = "Focus  t = $(round(times[i]/3600, digits=1)) h",
            xlabel       = "", ylabel = "",
            xticks       = false, yticks = false,
            colorbar     = false,
            legend       = false,
            xlims        = (-1.6, 1.6), ylims = (-1.6, 1.6),
            size         = frame_size,
        )
        for (gx, gy) in grat_dyn
            plt.plot!(p_globe, gx, gy, color = :grey60, lw = 0.8, label = false)
        end
        plt.plot!(p_globe, eq_dyn[1], eq_dyn[2], color = :grey30, lw = 1.5, label = false)
        # Depth-aware axis: exterior stubs always visible; interior only for the near-side pole.
        let _uz = cos(cam_el_dyn * π/180), _top = 1.5 * cos(cam_el_dyn * π/180)
            plt.plot!(p_globe, [0.0, 0.0], [ 1.0,  _top], color=:grey30, lw=1.5, ls=:solid, label=false)
            plt.plot!(p_globe, [0.0, 0.0], [-1.0, -_top], color=:grey30, lw=1.5, ls=:solid, label=false)
            if cam_el_dyn > 1.0        # north pole faces camera → north interior on top
                plt.plot!(p_globe, [0.0, 0.0], [_uz, 1.0],  color=:grey30, lw=1.5, ls=:solid, label=false)
            elseif cam_el_dyn < -1.0   # south pole faces camera → south interior on top
                plt.plot!(p_globe, [0.0, 0.0], [-_uz, -1.0], color=:grey30, lw=1.5, ls=:solid, label=false)
            end
        end
        plt.plot!(p_globe, cos.(θ_c), sin.(θ_c), color = :black, lw = 2, label = false)
    else
        # Two fixed half-globes (front + back)
        img, s_px  = ortho_image(fstate, data_lon, data_lat, cam_az, cam_el; N=300)
        img[land_visible1] .= LAND_SENTINEL_E

        p_map = plt.heatmap(
            s_px, s_px, img',
            color        = land_energy_cgrad,
            clims        = (LAND_SENTINEL_E, max_energy),
            aspect_ratio = :equal,
            title        = "Front  t = $(round(times[i]/3600, digits=1)) h",
            xlabel       = "", ylabel = "",
            xticks       = false, yticks = false,
            colorbar     = false,
            legend       = false,
            size         = frame_size,
        )
        for (gx, gy) in graticule1
            plt.plot!(p_map, gx, gy, color = :grey60, lw = 0.8, label = false)
        end
        plt.plot!(p_map, equator1[1], equator1[2], color = :grey30, lw = 1.5, label = false)
        plt.plot!(p_map, axis1_north[1], axis1_north[2], color = :grey30, lw = 1.5, ls = :solid, label = false)
        plt.plot!(p_map, axis1_south[1], axis1_south[2], color = :grey30, lw = 1.5, ls = :solid, label = false)
        plt.plot!(p_map, cos.(θ_c), sin.(θ_c), color = :black, lw = 2, label = false)

        img2, _ = ortho_image(fstate, data_lon, data_lat, cam_az2, cam_el; N=300)
        img2[land_visible2] .= LAND_SENTINEL_E

        p_map2 = plt.heatmap(
            s_px, s_px, img2',
            color        = land_energy_cgrad,
            clims        = (LAND_SENTINEL_E, max_energy),
            aspect_ratio = :equal,
            title        = "Back",
            xlabel       = "", ylabel = "",
            xticks       = false, yticks = false,
            colorbar     = false,
            legend       = false,
            size         = frame_size,
        )
        for (gx, gy) in graticule2
            plt.plot!(p_map2, gx, gy, color = :grey60, lw = 0.8, label = false)
        end
        plt.plot!(p_map2, equator2[1], equator2[2], color = :grey30, lw = 1.5, label = false)
        plt.plot!(p_map2, axis2_north[1], axis2_north[2], color = :grey30, lw = 1.5, ls = :solid, label = false)
        plt.plot!(p_map2, axis2_south[1], axis2_south[2], color = :grey30, lw = 1.5, ls = :solid, label = false)
        plt.plot!(p_map2, cos.(θ_c), sin.(θ_c), color = :black, lw = 2, label = false)
    end

    # ------------------------------------------------------------------
    # Left panel: time series
    # ------------------------------------------------------------------

    # Full series in grey/light colour (background reference)
    p_ts = plt.plot(
        times[1:end] ./ 3600,
        [total_energies ones(iterations) .* total_energies[1]],
        label   = false,
        xlims   = (times[1]/3600, times[end]/3600),
        ylims   = (0, 1.5 * max_total_energy),
        linewidth = [1 2],
        ls      = [:dot :solid],
        color   = ["#ef8b8b" :purple],
    )

    # Highlighted series up to current frame
    plt.plot!(
        times[1:i] ./ 3600,
        [total_energies[1:i] total_energies[1:i]],
        label   = false,
        ylabel  = "Total energy in domain (m² · cells)",
        xlims   = (times[1]/3600, times[end]/3600),
        ylims   = (0, 1.5 * max_total_energy),
        linewidth = [3 3],
        ls      = [:solid :dot],
        color   = [:white "#c13030"],
    )

    # Max wave speed on the right y-axis
    y_max_speed = 1.3 * maximum(max_speeds)

    plt.plot!(
        plt.twinx(),
        times[1:end] ./ 3600,
        max_speeds,
        label     = false,
        title     = "Wave speed and energy",
        ylabel    = "Speed (m/s)",
        xlabel    = "Time (hours)",
        xlims     = (times[1]/3600, times[end]/3600),
        ylims     = (0, y_max_speed),
        linewidth = 2,
        ls        = :solid,
        color     = "#ffcf9c",
    )

    plt.plot!(
        plt.twinx(),
        times[1:i] ./ 3600,
        max_speeds[1:i],
        label     = "Max wave speed (m/s)",
        ylabel    = "Speed (m/s)",
        xlims     = (times[1]/3600, times[end]/3600),
        ylims     = (0, y_max_speed),
        linewidth = 4,
        ls        = :solid,
        color     = "#ff8300",
    )

    # ------------------------------------------------------------------
    # Compose and save
    # ------------------------------------------------------------------
    if focus_globe
        l_e     = @plt.layout [a{0.15w} b{0.77w} c{0.08w}]
        p_final = plt.plot(p_ts, p_globe, p_cb,
            layout = l_e, size = frame_size, margin = 15plt.mm)
    else
        l_e     = @plt.layout [a{0.18w} b{0.37w} c{0.37w} d{0.08w}]
        p_final = plt.plot(p_ts, p_map, p_map2, p_cb,
            layout = l_e, size = frame_size, margin = 15plt.mm)
    end

    plt.savefig(p_final, data_path * "/heatmaps/" * string(i) * ".png")

end

println("")
println("Frames saved to: " * data_path * "/heatmaps/")
println("")

create_movie(data_path * "/heatmaps", data_path * "/sphere_movie.mp4"; framerate)
println("")


# ============================================================================
# 4. Wave speed globe plots
# ============================================================================

mkpath(data_path * "/wave_speed")

# Energy threshold below which wave speed is not shown
E_threshold_ws = max_energy * E_threshold_ws_frac

# Sparse arrow grid (spacing set by N_skip_arr in user parameters)
i_arr = 1:N_skip_arr:Nlon
j_arr = 1:N_skip_arr:Nlat

# Global max group speed for a fixed colorbar across all frames
ws_all_flat = sqrt.(particle_data[:, :, 2].^2 .+ particle_data[:, :, 3].^2)
max_ws_global = maximum(ws_all_flat[(particle_data[:, :, 1] .> log(E_threshold_ws)) .& isfinite.(ws_all_flat)])

ws_cgrad = plt.cgrad(:plasma)

# Standalone colorbar strip for wave speed (static — computed once)
min_ws = minimum(ws_all_flat[(particle_data[:, :, 1] .> log(E_threshold_ws)) .& isfinite.(ws_all_flat) .& (ws_all_flat .> 0)])

# Wave-speed land sentinel + colormap (needs min_ws / max_ws_global — defined just above)
LAND_SENTINEL_WS = min_ws - (max_ws_global - min_ws)
_ws_base_cgrad   = plt.cgrad(:plasma)
ws_ocean_cols    = [_ws_base_cgrad[p] for p in c_pts]
ws_land_cgrad    = plt.cgrad(vcat(fill(saddlebrown, 2), ws_ocean_cols), full_pts)

ws_cb_axis = collect(range(min_ws, max_ws_global, length=256))
p_cb_ws = plt.heatmap(
    [0.0], ws_cb_axis, reshape(ws_cb_axis, 1, :),
    color        = ws_cgrad,
    clims        = (min_ws, max_ws_global),
    colorbar     = false,
    xticks       = false,
    yaxis        = :right,
    ylabel       = "Group speed [m/s]",
    framestyle   = :box,
    aspect_ratio = :auto,
)

N_probes = length(probe_lons)

# Pre-compute nearest grid indices for each probe (done once, reused every frame)
probe_i_lon = [argmin(abs.(data_lon .- lon_p)) for lon_p in probe_lons]
probe_j_lat = [argmin(abs.(data_lat .- lat_p)) for lat_p in probe_lats]
probe_idx   = [probe_i_lon[k] + Nlon * (probe_j_lat[k] - 1) for k in 1:N_probes]

# Velocity-space range for the spectrum heatmaps
spec_c_max = max_ws_global * spec_c_max_frac
spec_range = range(-spec_c_max, spec_c_max, length=N_spec)


"""
    draw_speed_arrows!(p, xs, ys, us, vs; head_len, head_width, color, lw)

Draw direction arrows on plot `p` with properly sized arrowheads. Arrowheads
are rendered as filled triangles via `plt.Shape` — the only approach that
gives exact control over size in data coordinates with the GR backend.
All shafts are drawn in one batched `plot!` call; all arrowheads in one
`Shape` call, so cost is O(1) plot calls regardless of arrow count.
"""
function draw_speed_arrows!(p, xs, ys, us, vs;
                            head_len  = 0.026,
                            head_width = 0.016,
                            color = :black,
                            lw    = 1.0)
    isempty(xs) && return
    shaft_x = Float64[];  shaft_y = Float64[]
    head_x  = Float64[];  head_y  = Float64[]
    for k in eachindex(xs)
        u, v = us[k], vs[k]
        s    = sqrt(u^2 + v^2)
        s < 1e-10 && continue
        ux, uy = u/s, v/s      # unit direction
        nx, ny = -uy, ux       # left perpendicular

        # Shaft: tail → arrowhead base
        bx = xs[k] + u - head_len * ux
        by = ys[k] + v - head_len * uy
        push!(shaft_x, xs[k], bx, NaN)
        push!(shaft_y, ys[k], by, NaN)

        # Arrowhead triangle: tip, left-base, right-base
        tx, ty = xs[k] + u, ys[k] + v
        push!(head_x, tx, bx + head_width*nx, bx - head_width*nx, NaN)
        push!(head_y, ty, by + head_width*ny, by - head_width*ny, NaN)
    end
    plt.plot!(p, shaft_x, shaft_y, color=color, lw=lw, label=false)
    plt.plot!(p, plt.Shape(head_x, head_y), fillcolor=color, linewidth=0, label=false)
end

"""
    make_spectrum_panel(cx_p, cy_p, Σ_kk, spec_range, spec_c_max, spec_cgrad; title_str)

2D velocity-space wave spectrum at a probe location.
The spectrum is a 2D Gaussian centered at (cx_p, cy_p) with covariance Σ_kk
(the velocity-velocity block of the particle covariance matrix).
Overlays concentric speed circles and radial direction lines.
"""
function make_spectrum_panel(cx_p, cy_p, Σ_kk, spec_range, spec_c_max, spec_cgrad, energy, max_energy, min_energy;
                              title_str = "", probe_color = :white, active = true)
    N = length(spec_range)
    spec = fill(NaN, N, N)
    # Fill inside circle with background level (makes inactive panels visible as a dark disk)
    for (jj, cy) in enumerate(spec_range)
        for (ii, cx) in enumerate(spec_range)
            sqrt(cx^2 + cy^2) <= spec_c_max && (spec[ii, jj] = min_energy)
        end
    end
    if active && energy >= min_energy + 3
        σ_floor = spec_c_max / 8.0
        Σ_display = Σ_kk + [σ_floor^2  0.0; 0.0  σ_floor^2]
        Σ_inv = try
            inv(Σ_display)
        catch
            [1.0/σ_floor^2  0.0; 0.0  1.0/σ_floor^2]
        end
        for (jj, cy) in enumerate(spec_range)
            for (ii, cx) in enumerate(spec_range)
                sqrt(cx^2 + cy^2) > spec_c_max && continue
                d = [cx - cx_p, cy - cy_p]
                spec[ii, jj] = log(exp(-0.5 * (d' * Σ_inv * d)) * exp(energy))
            end
        end
    end

    cr = collect(spec_range)
    p = plt.heatmap(cr, cr, spec',
        color          = spec_cgrad,
        colorbar       = false,
        aspect_ratio   = :equal,
        title          = title_str,
        titlefontcolor = probe_color,
        clims          = (min_energy + 3, max_energy - 1),
        framestyle     = :none,
        xlims          = (-spec_c_max * 1.12, spec_c_max * 1.12),
        ylims          = (-spec_c_max * 1.12, spec_c_max * 1.12),
        xticks         = false,
        yticks         = false,
    )

    # Concentric speed circles
    θ_circ = range(0, 2π, length=200)
    for r_frac in [0.25, 0.5, 0.75, 1.0]
        r = spec_c_max * r_frac
        plt.plot!(p, r .* cos.(θ_circ), r .* sin.(θ_circ),
                  color = :grey70, lw = 0.8, ls = :dash, label = false)
    end

    # Radial direction lines every 45°
    for θ_deg in 0:45:315
        θ_r = θ_deg * π / 180
        plt.plot!(p, [0.0, spec_c_max * cos(θ_r)], [0.0, spec_c_max * sin(θ_r)],
                  color = :grey70, lw = 0.8, ls = :dot, label = false)
    end

    # Direction labels (N/E/S/W) just inside the circle perimeter
    label_r = spec_c_max * 0.88
    for (θ_deg, lbl) in zip([90, 0, 270, 180], ["N", "E", "S", "W"])
        θ_r = θ_deg * π / 180
        plt.annotate!(p, label_r * cos(θ_r), label_r * sin(θ_r),
                      plt.text(lbl, :white, :center, 8))
    end

    # Speed labels along the NE radial (45°), one per inner ring
    label_θ = π / 4
    for r_frac in [0.25, 0.5, 0.75]
        r = spec_c_max * r_frac
        speed_val = round(r; digits = 1)
        plt.annotate!(p, r * cos(label_θ), r * sin(label_θ),
                      plt.text("$(speed_val)", :grey80, :left, 6))
    end

    # Mark mean velocity with a white cross
    plt.scatter!(p, [cx_p], [cy_p], color = :white, ms = 6,
                 markershape = :xcross, markerstrokewidth = 2,
                 markerstrokecolor = :white, label = false)

    return p
end


# Flat ocean mask: true where the grid cell is ocean (reuses the land_field computed above).
# land_field[i,j] = 1.0 on land; vec() preserves the same i + Nlon*(j-1) flat ordering
# as the particle_data second axis.
ocean_flat = vec(land_field) .< 0.5

# Pre-compute the global maximum energy-weighted histogram bin so that the
# x-axis stays fixed across all frames.
hist_bw      = (max_ws_global - min_ws) / n_hist_bins
max_hist_val = 0.0
for j in 1:iterations
    lne_j   = particle_data[j, :, 1]
    ws_flat = sqrt.(particle_data[j, :, 2].^2 .+ particle_data[j, :, 3].^2)
    mask_j  = (lne_j .> log(E_threshold_ws)) .& isfinite.(ws_flat) .& ocean_flat
    any(mask_j) || continue
    ws_j = sqrt.(particle_data[j, :, 2].^2 .+ particle_data[j, :, 3].^2)[mask_j]
    e_j  = exp.(lne_j[mask_j])
    h = zeros(n_hist_bins)
    for (s, w) in zip(ws_j, e_j)
        k = clamp(ceil(Int, (s - min_ws) / hist_bw), 1, n_hist_bins)
        h[k] += w
    end
    global max_hist_val = max(max_hist_val, maximum(h))
end
max_hist_val *= 0.5   # add some headroom above the tallest bin

# Two independent row layouts combined via ffmpeg vstack
l_top = focus_globe ?
    (@plt.layout [a{0.77w} b{0.08w} c{0.15w}]) :
    (@plt.layout [a{0.33w} b{0.33w} c{0.09w} d{0.25w}])
l_bot = @plt.layout [a b c d]

tmp_top = data_path * "/wave_speed/_tmp_top.png"
tmp_bot = data_path * "/wave_speed/_tmp_bot.png"

for i in 1:iterations

    if i % 10 == 0
        println("Wave speed frame " * string(i) * " / " * string(iterations))
    end

    fstate   = mesh_data[i, :, :]   # (Nlon × Nlat) energy field

    # Group speed field; mask cells where energy is below the threshold
    ws_flat  = sqrt.(particle_data[i, :, 2].^2 .+ particle_data[i, :, 3].^2)
    ws_field = reshape(ws_flat, Nlon, Nlat)
    ws_field[fstate .< E_threshold_ws] .= NaN

    # Direction arrows: use raw momentum (m_x, m_y) — same direction as (c̄_x, c̄_y)
    cx_field  = reshape(particle_data[i, :, 2], Nlon, Nlat)
    cy_field  = reshape(particle_data[i, :, 3], Nlon, Nlat)
    # Use per-particle log-energy (column 1) for the threshold: background particles
    # carry lne = -10.87 (≪ 0 ≈ 1e-9), so the filter correctly removes dead cells.
    lne_field = reshape(particle_data[i, :, 1], Nlon, Nlat)

    arr1_x, arr1_y, arr1_u, arr1_v = project_arrows(
        cx_field, cy_field, lne_field, data_lon, data_lat,
        i_arr, j_arr, cam_az, cam_el;
        E_threshold = log(E_threshold_ws), arrow_scale = arrow_scale_ws)

    arr2_x, arr2_y, arr2_u, arr2_v = project_arrows(
        cx_field, cy_field, lne_field, data_lon, data_lat,
        i_arr, j_arr, cam_az2, cam_el;
        E_threshold = log(E_threshold_ws), arrow_scale = arrow_scale_ws)

    # Energy-weighted speed histogram (active ocean particles only)
    lne_all     = particle_data[i, :, 1]
    active_mask = (lne_all .> log(E_threshold_ws)) .& isfinite.(ws_flat) .& ocean_flat
    ws_hist     = ws_flat[active_mask]
    e_hist      = exp.(lne_all[active_mask])

    if !isempty(ws_hist)
        avg_speed = sum(ws_hist .* e_hist) / sum(e_hist)

        p_hist = plt.histogram(ws_hist,
            weights     = e_hist,
            bins        = range(min_ws, max_ws_global, length=n_hist_bins+1),
            orientation = :horizontal,
            color       = "#e67e22",
            label       = false,
            title       = "Speed distribution",
            xlabel      = "Energy [m²]",
            ylabel      = "Group speed [m/s]",
            xlims       = (0, max_hist_val),
            ylims       = (min_ws, max_ws_global),
        )
        plt.hline!(p_hist, [avg_speed],
            color     = :red,
            linewidth = 4,
            ls        = :dash,
            label     = false)
    else
        p_hist = plt.plot(title = "Speed distribution", label = false,
                          xlims = (0, max_hist_val),
                          ylims = (min_ws, max_ws_global))
    end

    # Probe N_probes tracks the peak ocean-energy cell for this frame
    let peak_ci = argmax(fstate .* (1.0 .- land_field))
        probe_i_lon[N_probes] = peak_ci[1]
        probe_j_lat[N_probes] = peak_ci[2]
        probe_idx[N_probes]   = probe_i_lon[N_probes] + Nlon * (probe_j_lat[N_probes] - 1)
        probe_lons[N_probes]  = data_lon[probe_i_lon[N_probes]]
        probe_lats[N_probes]  = data_lat[probe_j_lat[N_probes]]
    end

    # 2D wave spectra at probe locations
    log_max_energy = (maximum([maximum(particle_data[:, probe_idx[i],1]) for i in 1:4]))
    log_min_energy = (minimum([minimum(particle_data[:, probe_idx[i],1]) for i in 1:4]))
    spec_panels = Vector{Any}(undef, N_probes)
    for k in 1:N_probes
        p_idx  = probe_idx[k]
        cx_p   = particle_data[i, p_idx, 2]
        cy_p   = particle_data[i, p_idx, 3]

        lon_lab = round(Int, probe_lons[k])
        lat_lab = round(Int, probe_lats[k])
        title_str = "Log-Spectrum @ ($(lon_lab)°E, $(lat_lab)°N)"

        mesh_energy = fstate[probe_i_lon[k], probe_j_lat[k]]
        cov_vec = particle_data[i, p_idx, 6:15]
        Σ_kk    = fold(cov_vec)[1:2, 1:2]
        # print("Probe $k: energy=$(round(particle_data[i, p_idx, 1], sigdigits=3)), cx=$(round(cx_p, sigdigits=3)), cy=$(round(cy_p, sigdigits=3)), max_energy=$(round(log_max_energy, sigdigits=3)), min_energy=$(round(log_min_energy, sigdigits=3))\n")
        spec_panels[k] = make_spectrum_panel(cx_p, cy_p, Σ_kk, spec_range,
                                              spec_c_max, spec_cgrad, particle_data[i, p_idx, 1], log_max_energy, log_min_energy;
                                              title_str   = title_str,
                                              probe_color = probe_colors[k],
                                              active      = mesh_energy >= E_threshold_ws)
    end

    if focus_globe
        # Single focus globe: use pre-smoothed camera path
        cam_az_dyn = cam_az_focus[i]
        cam_el_dyn = cam_el_focus[i]

        grat_dyn_ws = [project_meridian(λ, cam_az_dyn, cam_el_dyn) for λ in lon_lines]
        eq_dyn_ws   = project_parallel(0.0, cam_az_dyn, cam_el_dyn)

        lnd_dyn_ws, _ = ortho_image(land_field, data_lon, data_lat, cam_az_dyn, cam_el_dyn; N=300)
        lvis_dyn_ws   = (.!isnan.(lnd_dyn_ws)) .& (lnd_dyn_ws .> 0.5)

        arr_dyn_x, arr_dyn_y, arr_dyn_u, arr_dyn_v = project_arrows(
            cx_field, cy_field, lne_field, data_lon, data_lat,
            i_arr, j_arr, cam_az_dyn, cam_el_dyn;
            E_threshold = log(E_threshold_ws), arrow_scale = arrow_scale_ws)

        ws_img_dyn, s_px_ws = ortho_image(ws_field, data_lon, data_lat, cam_az_dyn, cam_el_dyn; N=300)
        ws_img_dyn[lvis_dyn_ws] .= LAND_SENTINEL_WS

        p_ws_globe = plt.heatmap(
            s_px_ws, s_px_ws, ws_img_dyn',
            color        = ws_land_cgrad,
            clims        = (LAND_SENTINEL_WS, max_ws_global),
            aspect_ratio = :equal,
            title        = "Group speed (focus)  t = $(round(times[i]/3600, digits=1)) h",
            xlabel       = "", ylabel = "",
            xticks       = false, yticks = false,
            colorbar     = false,
            legend       = false,
            xlims        = (-1.6, 1.6), ylims = (-1.6, 1.6),
            size         = frame_size,
        )
        for (gx, gy) in grat_dyn_ws
            plt.plot!(p_ws_globe, gx, gy, color = :grey60, lw = 0.8, label = false)
        end
        plt.plot!(p_ws_globe, eq_dyn_ws[1], eq_dyn_ws[2], color = :grey30, lw = 1.5, label = false)
        # Depth-aware axis: exterior stubs always visible; interior only for the near-side pole.
        let _uz = cos(cam_el_dyn * π/180), _top = 1.5 * cos(cam_el_dyn * π/180)
            plt.plot!(p_ws_globe, [0.0, 0.0], [ 1.0,  _top], color=:grey30, lw=1.5, ls=:solid, label=false)
            plt.plot!(p_ws_globe, [0.0, 0.0], [-1.0, -_top], color=:grey30, lw=1.5, ls=:solid, label=false)
            if cam_el_dyn > 1.0        # north pole faces camera → north interior on top
                plt.plot!(p_ws_globe, [0.0, 0.0], [_uz, 1.0],  color=:grey30, lw=1.5, ls=:solid, label=false)
            elseif cam_el_dyn < -1.0   # south pole faces camera → south interior on top
                plt.plot!(p_ws_globe, [0.0, 0.0], [-_uz, -1.0], color=:grey30, lw=1.5, ls=:solid, label=false)
            end
        end
        draw_speed_arrows!(p_ws_globe, arr_dyn_x, arr_dyn_y, arr_dyn_u, arr_dyn_v)
        for k in 1:N_probes
            pt = project_point_to_screen(probe_lons[k], probe_lats[k], cam_az_dyn, cam_el_dyn)
            pt !== nothing && plt.scatter!(p_ws_globe, [pt[1]], [pt[2]],
                color = probe_colors[k], ms = 10, markershape = :circle,
                markerstrokecolor = :black, markerstrokewidth = 1.5, label = false)
        end
        plt.plot!(p_ws_globe, cos.(θ_c), sin.(θ_c), color = :black, lw = 2, label = false)

        p_top = plt.plot(p_ws_globe, p_cb_ws, p_hist,
            layout = l_top, size = (1920, 800),
            top_margin = 15plt.mm, left_margin = 15plt.mm, right_margin = 15plt.mm,
            bottom_margin = 5plt.mm)
    else
        # Two fixed half-globes (front + back)
        ws_img, s_px_ws = ortho_image(ws_field, data_lon, data_lat, cam_az, cam_el; N=300)
        ws_img[land_visible1] .= LAND_SENTINEL_WS

        p_ws1 = plt.heatmap(
            s_px_ws, s_px_ws, ws_img',
            color        = ws_land_cgrad,
            clims        = (LAND_SENTINEL_WS, max_ws_global),
            aspect_ratio = :equal,
            title        = "Group speed (front)  t = $(round(times[i]/3600, digits=1)) h",
            xlabel       = "", ylabel = "",
            xticks       = false, yticks = false,
            colorbar     = false,
            legend       = false,
            size         = frame_size,
        )
        for (gx, gy) in graticule1
            plt.plot!(p_ws1, gx, gy, color = :grey60, lw = 0.8, label = false)
        end
        plt.plot!(p_ws1, equator1[1], equator1[2], color = :grey30, lw = 1.5, label = false)
        plt.plot!(p_ws1, axis1_north[1], axis1_north[2], color = :grey30, lw = 1.5, ls = :solid, label = false)
        plt.plot!(p_ws1, axis1_south[1], axis1_south[2], color = :grey30, lw = 1.5, ls = :solid, label = false)
        draw_speed_arrows!(p_ws1, arr1_x, arr1_y, arr1_u, arr1_v)
        for k in 1:N_probes
            pt = project_point_to_screen(probe_lons[k], probe_lats[k], cam_az, cam_el)
            pt !== nothing && plt.scatter!(p_ws1, [pt[1]], [pt[2]],
                color = probe_colors[k], ms = 10, markershape = :circle,
                markerstrokecolor = :black, markerstrokewidth = 1.5, label = false)
        end
        plt.plot!(p_ws1, cos.(θ_c), sin.(θ_c), color = :black, lw = 2, label = false)

        ws_img2, _ = ortho_image(ws_field, data_lon, data_lat, cam_az2, cam_el; N=300)
        ws_img2[land_visible2] .= LAND_SENTINEL_WS

        p_ws2 = plt.heatmap(
            s_px_ws, s_px_ws, ws_img2',
            color        = ws_land_cgrad,
            clims        = (LAND_SENTINEL_WS, max_ws_global),
            aspect_ratio = :equal,
            title        = "Back",
            xlabel       = "", ylabel = "",
            xticks       = false, yticks = false,
            colorbar     = false,
            legend       = false,
            size         = frame_size,
        )
        for (gx, gy) in graticule2
            plt.plot!(p_ws2, gx, gy, color = :grey60, lw = 0.8, label = false)
        end
        plt.plot!(p_ws2, equator2[1], equator2[2], color = :grey30, lw = 1.5, label = false)
        plt.plot!(p_ws2, axis2_north[1], axis2_north[2], color = :grey30, lw = 1.5, ls = :solid, label = false)
        plt.plot!(p_ws2, axis2_south[1], axis2_south[2], color = :grey30, lw = 1.5, ls = :solid, label = false)
        draw_speed_arrows!(p_ws2, arr2_x, arr2_y, arr2_u, arr2_v)
        for k in 1:N_probes
            pt = project_point_to_screen(probe_lons[k], probe_lats[k], cam_az2, cam_el)
            pt !== nothing && plt.scatter!(p_ws2, [pt[1]], [pt[2]],
                color = probe_colors[k], ms = 10, markershape = :circle,
                markerstrokecolor = :black, markerstrokewidth = 1.5, label = false)
        end
        plt.plot!(p_ws2, cos.(θ_c), sin.(θ_c), color = :black, lw = 2, label = false)

        p_top = plt.plot(p_ws1, p_ws2, p_cb_ws, p_hist,
            layout = l_top, size = (1920, 800),
            top_margin = 15plt.mm, left_margin = 15plt.mm, right_margin = 15plt.mm,
            bottom_margin = 5plt.mm)
    end
    plt.savefig(p_top, tmp_top)

    p_bot = plt.plot(spec_panels[1], spec_panels[2], spec_panels[3], spec_panels[4],
        layout = l_bot, size = (1920, 720), margin = 1plt.mm)
    plt.savefig(p_bot, tmp_bot)

    out_path = data_path * "/wave_speed/" * string(i) * ".png"
    run(`$(FFMPEG_jll.ffmpeg()) -y -loglevel error
        -i $tmp_top -i $tmp_bot
        -filter_complex "[0:v][1:v]vstack=inputs=2"
        $out_path`)

end

isfile(tmp_top) && rm(tmp_top)
isfile(tmp_bot) && rm(tmp_bot)

println("")
println("Wave speed frames saved to: " * data_path * "/wave_speed/")
println("")

create_movie(data_path * "/wave_speed", data_path * "/wave_speed_movie.mp4"; framerate)
println("")



# # ============================================================================
# # 4. Covariance matrix diagnostics along the great-circle swell track
# #
# #    On a spherical grid there is no natural y=x diagonal.  Instead we extract
# #    covariance data along a constant-latitude band centred on the storm's
# #    latitude, traversing eastward from the storm longitude. This is the
# #    great-circle path of the dominant swell beam for a westerly-propagating
# #    storm wind source.
# #
# #    Mirrors the commented-out Cartesian covariance block.
# # ============================================================================

# # Storm centre (must match values in test_case_parametric_spherical.jl)
# storm_lon = 320.0   # °E
# storm_lat =  40.0   # °N

# # Find the latitude row and starting longitude column nearest the storm
# storm_lat_idx = argmin(abs.(data_lat .- storm_lat))
# storm_lon_idx = argmin(abs.(data_lon .- storm_lon))

# # Distance from the storm along the constant-latitude great-circle (km)
# # dx at storm latitude in km
# R_earth_km   = 6371.0
# dx_at_lat_km = dlon * π / 180.0 * R_earth_km * cosd(storm_lat)
# dy_km        = dlat * π / 180.0 * R_earth_km

# # East-of-storm longitude indices (wrapping around 360° if necessary)
# east_indices = [(storm_lon_idx - 1 + k - 1) % Nlon + 1 for k in 1:Nlon]

# # Great-circle distances in km from the storm along the lat band
# distances_km = [(k - 1) * dx_at_lat_km for k in 1:Nlon]

# # Output arrays for the final iteration (for the log-log plot)
# end_cov_cxcx = zeros(Nlon)
# end_cov_cxcy = zeros(Nlon)
# end_cov_cycy = zeros(Nlon)
# end_cov_xx   = zeros(Nlon)

# for i in 1:iterations

#     cov_sum  = zeros(Nlon)
#     cov_cxcx = zeros(Nlon)
#     cov_cxcy = zeros(Nlon)
#     cov_cycy = zeros(Nlon)
#     cov_cxx  = zeros(Nlon)
#     cov_cxy  = zeros(Nlon)
#     cov_cyx  = zeros(Nlon)
#     cov_cyy  = zeros(Nlon)
#     cov_xx   = zeros(Nlon)
#     cov_xy   = zeros(Nlon)
#     cov_yy   = zeros(Nlon)

#     fstate = mesh_data[i, :, :]   # (Nlon × Nlat)

#     # Extract covariance along the storm-latitude band, heading east
#     for (k, lon_idx) in enumerate(east_indices)
#         j = storm_lat_idx    # constant latitude row

#         energy     = fstate[lon_idx, j]
#         moment_amp = sqrt(particle_data[i, lon_idx + Nlon*(j-1), 2]^2 +
#                           particle_data[i, lon_idx + Nlon*(j-1), 3]^2)

#         if energy < 1e-8 || moment_amp < 1e-12
#             continue
#         end

#         # Recover the covariance matrix from the particle state vector
#         cov_flat = particle_data[i, lon_idx + Nlon*(j-1), 6:15]
#         m_cov    = fold(cov_flat) .* energy ./ (2 * moment_amp^2)

#         cov_sum[k]  = sum(m_cov[1:2, 1:2])
#         cov_cxcx[k] = m_cov[1, 1]
#         cov_cxcy[k] = m_cov[1, 2]
#         cov_cycy[k] = m_cov[2, 2]
#         cov_cxx[k]  = m_cov[1, 3]
#         cov_cxy[k]  = m_cov[1, 4]
#         cov_cyx[k]  = m_cov[2, 3]
#         cov_cyy[k]  = m_cov[2, 4]
#         cov_xx[k]   = m_cov[3, 3]
#         cov_xy[k]   = m_cov[3, 4]
#         cov_yy[k]   = m_cov[4, 4]
#     end

#     # ---- Plot: covariance subblocks along the swell track ----------------
#     plt.plot(
#         distances_km, cov_sum,
#         title   = "Cov. matrix elements along swell track   (lat ≈ $(round(data_lat[storm_lat_idx], digits=1))°N, t = $(round(times[i]/3600, digits=1)) h)",
#         label   = "Tr(P_cc) = cov(c_x,c_x) + cov(c_y,c_y)",
#         legend  = :topleft,
#         xlabel  = "Distance east of storm (km)",
#         ylabel  = "Covariance (m²/s²)",
#         ylims   = (0, max(1.0, 1.1 * maximum(cov_sum[isfinite.(cov_sum)]))),
#         size    = (1200, 600),
#         linewidth = 3,
#         color   = :blue,
#     )
#     plt.plot!(distances_km, cov_cxcx, label = "cov(c_x, c_x)", color = :red,   linewidth = 2)
#     plt.plot!(distances_km, cov_cxcy, label = "cov(c_x, c_y)", color = :green, linewidth = 2)
#     plt.plot!(distances_km, cov_cycy, label = "cov(c_y, c_y)", color = :orange,linewidth = 2)

#     plt.savefig(data_path * "/covariances/cov_cc_" * string(i) * ".png")

#     # ---- Plot: cross-covariance (c,x) subblock ---------------------------
#     # plt.plot(distances_km, cov_cxx, label = "cov(c_x, x)", ...)
#     # plt.savefig(...)

#     # Save the final-iteration arrays for the log-log plot
#     if i == iterations
#         end_cov_xx   = cov_xx
#         end_cov_cxcx = cov_cxcx
#         end_cov_cxcy = cov_cxcy
#         end_cov_cycy = cov_cycy
#     end

# end


# # ============================================================================
# # 5. Log-log plot: Tr(P_cc) vs. propagation distance
# #
# #    Under the Lyapunov equation for free swell, P_cc is frozen (constant),
# #    so Tr(P_cc) should plateau after the swell leaves the generating region.
# #    If random current effects are included, Tr(P_cc) ~ t ~ distance, giving
# #    a slope of 1 on the log-log plot.
# #
# #    P_xx should grow as distance² (slope 2 on log-log): the spatial variance
# #    of the wave packet grows because directionally spread sub-packets separate
# #    at a rate proportional to the directional spread.
# # ============================================================================

# # Find the first index where swell has clearly arrived (non-zero cov_cxcx)
# energy_threshold    = 1e-4
# swell_mask          = end_cov_cxcx .> energy_threshold
# first_swell_index   = findfirst(swell_mask)

# if !isnothing(first_swell_index) && first_swell_index < Nlon - 5

#     swell_range = first_swell_index:Nlon
#     swell_cov   = end_cov_cxcx[swell_range]
#     swell_dist  = distances_km[swell_range]
#     swell_xx    = end_cov_xx[swell_range]

#     # Remove zeros and negatives before log
#     valid = (swell_cov .> 0) .& (swell_dist .> 0) .& (swell_xx .> 0)

#     # Reference lines: slope 0 (frozen P_cc) and slope 2 (growing P_xx)
#     d_ref   = swell_dist[valid]
#     d_norm  = d_ref ./ d_ref[1]
#     ref0    = ones(sum(valid)) .* mean(log10.(swell_cov[valid]))   # flat reference
#     ref2    = log10.(swell_cov[valid][1]) .+ 2 .* log10.(d_norm)  # slope 2

#     plt.plot(
#         swell_dist[valid], swell_cov[valid],
#         title     = "Log-log: directional variance Tr(P_cc) vs. propagation distance",
#         label     = "Tr(P_cc)  (simulation)",
#         xlabel    = "Propagation distance east of storm (km)",
#         ylabel    = "Tr(P_cc) = Var(c_x) + Var(c_y)   (m²/s²)",
#         xaxis     = :log10,
#         yaxis     = :log10,
#         size      = (1000, 700),
#         linewidth = 3,
#         color     = :royalblue,
#         legend    = :topleft,
#     )
#     plt.plot!(
#         10 .^ (range(log10(d_ref[1]), log10(d_ref[end]), length=50)),
#         10 .^ range(ref0[1], ref0[1], length=50),
#         label     = "slope 0 (frozen, expected for free swell)",
#         ls        = :dash,
#         color     = :green,
#         linewidth = 2,
#     )

#     plt.savefig(data_path * "/covariances/0_loglog_cov_cc_vs_distance.png")

#     # ---- Log-log for P_xx (spatial spread) ----------------------------
#     plt.plot(
#         swell_dist[valid], swell_xx[valid],
#         title     = "Log-log: spatial variance P_xx vs. propagation distance",
#         label     = "P_xx  (simulation)",
#         xlabel    = "Propagation distance east of storm (km)",
#         ylabel    = "P_xx = Var(x)   (m²)",
#         xaxis     = :log10,
#         yaxis     = :log10,
#         size      = (1000, 700),
#         linewidth = 3,
#         color     = :firebrick,
#         legend    = :topleft,
#     )
#     plt.plot!(
#         10 .^ (range(log10(d_ref[1]), log10(d_ref[end]), length=50)),
#         10 .^ (log10.(swell_xx[valid][1]) .+ 2 .* range(0, log10(d_ref[end]/d_ref[1]), length=50)),
#         label     = "slope 2 (expected: P_xx ∝ t²)",
#         ls        = :dash,
#         color     = :darkorange,
#         linewidth = 2,
#     )

#     plt.savefig(data_path * "/covariances/0_loglog_Pxx_vs_distance.png")

# else
#     println("Warning: swell not yet detected in final snapshot — log-log plot skipped.")
# end


# # ============================================================================
# # 6. Final-state global map of Tr(P_cc)
# #
# #    Shows the spatial pattern of directional variance at the end of the
# #    simulation on the spherical map — reveals the swell beam and its
# #    angular broadening as it propagates.
# # ============================================================================

# fstate_final = mesh_data[end, :, :]
# cov_cc_map   = zeros(Nlon, Nlat)

# for lon_idx in 1:Nlon, lat_idx in 1:Nlat
#     energy     = fstate_final[lon_idx, lat_idx]
#     flat_idx   = lon_idx + Nlon * (lat_idx - 1)
#     moment_amp = sqrt(particle_data[end, flat_idx, 2]^2 +
#                       particle_data[end, flat_idx, 3]^2)
#     if energy > 1e-8 && moment_amp > 1e-12
#         cov_flat       = particle_data[end, flat_idx, 6:15]
#         m_cov          = fold(cov_flat) .* energy ./ (2 * moment_amp^2)
#         cov_cc_map[lon_idx, lat_idx] = m_cov[1,1] + m_cov[2,2]   # Tr(P_cc)
#     end
# end

# plt.heatmap(
#     data_lon, data_lat, transpose(cov_cc_map),
#     proj         = :geo,
#     xlims        = (lon_min, lon_max),
#     ylims        = (lat_min, lat_max),
#     title        = "Final Tr(P_cc) = Var(c_x)+Var(c_y)  [m²/s²]   (t = $(round(times[end]/3600, digits=1)) h)",
#     xlabel       = "Longitude (°E)",
#     ylabel       = "Latitude (°N)",
#     color        = plt.cgrad(:viridis),
#     aspect_ratio = :equal,
#     size         = (1400, 700),
# )

# plt.savefig(data_path * "/covariances/final_Tr_Pcc_map.png")

# create_movie(data_path * "/covariances", data_path * "/covariances_movie.mp4";
#              framerate=8, pattern="cov_cc_%d.png")

# println("")
# println("Covariance plots saved to: " * data_path * "/covariances/")
# println("")
# println("Done !")