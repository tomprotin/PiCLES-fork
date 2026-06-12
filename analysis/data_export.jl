# ============================================================================
# data_export.jl  —  PiCLES → Godot binary export
#
# Converts CSV simulation output into Godot-friendly files.
# Run once after each simulation from the PiCLES project root:
#   julia analysis/data_export.jl
#
# Required packages (install once):
#   using Pkg; Pkg.add(["JSON3", "Images"])
# For geographic grids only (is_geographic = true):
#   using Pkg; Pkg.add("GeoDatasets")
#
# Output in <data_path>/godot_export/:
#   metadata.json        grid info, time array, binary layout
#   landmask.png         grayscale (Nlon × Nlat): white=land, black=ocean
#                        Row 0 = lat_max (north), last row = lat_min (south)
#   frames/NNNN.bin      per-timestep float32 binary, 4 sequential blocks:
#                        [ energy | lne | cx | cy ]
#                        each block is Nlon*Nlat float32 values,
#                        column-major: index = (lon_i - 1) + Nlon*(lat_j - 1)
# ============================================================================

# ── User parameters ──────────────────────────────────────────────────────────
data_path = pwd() * "/plots/test_case_parametric"
# data_path = pwd() * "/plots/good_test_cases/test_case_parametric"
# data_path = pwd() * "/plots/test_case_parametric_large_spectrum_7,5min_1000km_1000km"
out_path  = data_path * "/godot_export"

# Set to true  if coordinates are geographic (lon/lat in degrees, spherical globe).
# Set to false if coordinates are Cartesian (e.g. metres) — skips GeoDatasets lookup
# and writes an all-ocean landmask instead.
is_geographic = false
# ─────────────────────────────────────────────────────────────────────────────

using CSV, DataFrames, Printf
using JSON3
using Images
if is_geographic
    using GeoDatasets
end

println("\nPiCLES → Godot export")
println("  source : $data_path")
println("  output : $out_path\n")

mkpath(out_path)
mkpath(joinpath(out_path, "frames"))

# ============================================================================
# 1. Simulation metadata
# ============================================================================
println("Reading simulation metadata …")

meta_df    = CSV.read(joinpath(data_path, "data", "mesh_and_sim_data.csv"), DataFrame)
Nlon       = meta_df.Nx[1];      Nlat       = meta_df.Ny[1]
lon_min    = meta_df.xmin[1];    lon_max    = meta_df.xmax[1];   dlon = meta_df.dx[1]
lat_min    = meta_df.ymin[1];    lat_max    = meta_df.ymax[1];   dlat = meta_df.dy[1]
Δt         = meta_df.delta_t[1]; iterations = meta_df.n_iter[1]

data_lon   = collect(range(lon_min, lon_max, length=Nlon))
data_lat   = collect(range(lat_min, lat_max, length=Nlat))

winds_df   = CSV.read(joinpath(data_path, "data", "wind.csv"), DataFrame)
times      = Float32.(winds_df.t)

println("  Grid       : $(Nlon) × $(Nlat)")
println("  Lon range  : [$(lon_min)°, $(lon_max)°]")
println("  Lat range  : [$(lat_min)°, $(lat_max)°]")
println("  Iterations : $iterations")

# Sanity check: first particle CSV row count must match grid size
let df0 = CSV.read(
        joinpath(data_path, "data", "particles", "particles_1.csv"), DataFrame)
    n_rows = nrow(df0)
    n_expected = Nlon * Nlat
    n_rows == n_expected || error(
        "particles_1.csv has $n_rows rows but grid is $(Nlon)×$(Nlat)=$n_expected. " *
        "Delete old CSV files and re-run the simulation.")
end

# ============================================================================
# 2. Land mask → landmask.png
#    Row 0 = lat_max (north), matching standard map/texture orientation.
# ============================================================================
println("\nComputing land mask …")

if is_geographic
    lsm_lon_gd, lsm_lat_gd, lsm_data_gd = GeoDatasets.landseamask(resolution='c', grid=5)

    # land_field[i, j] = 1.0 where land, 0.0 where ocean  (Nlon × Nlat)
    land_field = [begin
        lo_w = mod(data_lon[i] + 180.0, 360.0) - 180.0
        ii   = argmin(abs.(lsm_lon_gd .- lo_w))
        jj   = argmin(abs.(lsm_lat_gd .- data_lat[j]))
        Float32(lsm_data_gd[ii, jj])
    end for i in 1:Nlon, j in 1:Nlat]

    # Transpose → (Nlat × Nlon), flip rows so row-0 = north (lat_max)
    # Explicitly use N0f8 (8-bit grayscale) — Godot's PNG loader rejects 16/32-bit PNGs
    land_img = Gray{N0f8}.(Float32.(land_field'[end:-1:1, :] .> 0.5))
else
    # Cartesian grid: coordinates are not geographic degrees, so GeoDatasets lookup
    # would map metres into random Earth locations. Use all-ocean mask instead.
    land_img = Gray{N0f8}.(zeros(Float32, Nlat, Nlon))
end

save(joinpath(out_path, "landmask.png"), land_img)
println("  → landmask.png  ($(Nlon) × $(Nlat) px, north at top)")

# ============================================================================
# 3. metadata.json
# ============================================================================
println("\nPre-scanning frames for global max energy …")
max_energy_global = 0.0f0
for i in 1:iterations
    (i == 1 || i % 100 == 0) && println("  Scanning frame $i / $iterations")
    global max_energy_global
    mesh_df = CSV.read(
        joinpath(data_path, "data", "mesh_values", "mesh_values_$(i).csv"), DataFrame)
    max_energy_global = max(max_energy_global, maximum(Float32.(vec(Matrix(mesh_df)'))))
end
println("  → max energy = $max_energy_global m²")


println("\nWriting metadata.json …")

frame_bytes = 7 * Nlon * Nlat * 4   # 7 fields × Nlon*Nlat × 4 bytes/float32

# Wave speed range computed during the binary writing loop below
max_ws_global = 0.0f0
min_ws_global = Inf32
E_threshold_ws = max_energy_global * 0.001f0   # matches energy_threshold_frac in shader

metadata = (;
    max_energy     = max_energy_global,
    max_wave_speed = 0.0,          # placeholder — patched after binary loop
    min_wave_speed = 0.0,          # placeholder
    grid_type      = is_geographic ? "geographic" : "cartesian",
    grid = (;
        Nlon, Nlat,
        lon_min = Float32(lon_min), lon_max = Float32(lon_max), dlon = Float32(dlon),
        lat_min = Float32(lat_min), lat_max = Float32(lat_max), dlat = Float32(dlat),
    ),
    simulation = (;
        iterations,
        delta_t = Float32(Δt),
        times   = times,
    ),
    frame_layout = (;
        dtype       = "float32",
        n_fields    = 7,
        field_names = ["energy", "lne", "cx", "cy", "cov_cxcx", "cov_cxcy", "cov_cycy"],
        field_size  = Nlon * Nlat,
        field_bytes = Nlon * Nlat * 4,
        description = "7 sequential float32 blocks. Column-major: index = (lon_i-1) + Nlon*(lat_j-1). cov_* are the velocity-velocity block of the 4x4 particle covariance matrix.",
    ),
    files = (;
        landmask    = "landmask.png",
        frame_fmt   = "frames/%04d.bin",
        frame_start = 1,
    ),
)

open(joinpath(out_path, "metadata.json"), "w") do f
    JSON3.pretty(f, metadata)
end
println("  → metadata.json")

# ============================================================================
# 4. Per-timestep binary frames
# ============================================================================
println("\nExporting $iterations frames  ($(round(frame_bytes / 1024, digits=1)) KB each) …\n")

for i in 1:iterations
    (i == 1 || i % 50 == 0) && println("  Frame $i / $iterations")

    # Energy — mesh CSV is (Nlat × Nlon); transpose → (Nlon × Nlat); flatten column-major
    mesh_df = CSV.read(
        joinpath(data_path, "data", "mesh_values", "mesh_values_$(i).csv"), DataFrame)
    energy = Float32.(vec(Matrix(mesh_df)'))

    # Particles — (Nlon*Nlat) rows; col 1 = row index, cols 2-15 = data
    part_df  = CSV.read(
        joinpath(data_path, "data", "particles", "particles_$(i).csv"), DataFrame)
    part_mat = Float32.(Matrix(part_df)[:, 2:end])

    lne      = part_mat[:, 1]   # log-energy
    cx       = part_mat[:, 2]   # eastward group velocity (m/s)
    cy       = part_mat[:, 3]   # northward group velocity (m/s)
    # Velocity-velocity block of the 4×4 covariance matrix (fold ordering: (1,1),(1,2),(2,2),…)
    cov_cxcx = part_mat[:, 6]   # Σ[1,1] = Var(cx)
    cov_cxcy = part_mat[:, 7]   # Σ[1,2] = Cov(cx,cy)
    cov_cycy = part_mat[:, 8]   # Σ[2,2] = Var(cy)

    open(joinpath(out_path, "frames", @sprintf("%04d.bin", i)), "w") do fh
        write(fh, energy)
        write(fh, lne)
        write(fh, cx)
        write(fh, cy)
        write(fh, cov_cxcx)
        write(fh, cov_cxcy)
        write(fh, cov_cycy)
    end

    # Accumulate wave speed range using the energy mask (matches shader threshold)
    ws     = sqrt.(cx .^ 2 .+ cy .^ 2)
    active = (energy .>= E_threshold_ws) .& (ws .> 0)
    if any(active)
        global max_ws_global, min_ws_global
        max_ws_global = max(max_ws_global, maximum(ws[active]))
        min_ws_global = min(min_ws_global, minimum(ws[active]))
    end
end

# Patch the metadata file with the computed wave speed range
min_ws_global = isfinite(min_ws_global) ? min_ws_global : 0.0f0
println("  Wave speed range: [$min_ws_global, $max_ws_global] m/s")
let meta_path = joinpath(out_path, "metadata.json")
    patched = Dict(pairs(JSON3.read(read(meta_path, String))))
    patched[:max_wave_speed] = max_ws_global
    patched[:min_wave_speed] = min_ws_global
    open(meta_path, "w") do f JSON3.pretty(f, patched) end
end

println("\nExport complete!")
println("  $(iterations) frames × $(round(frame_bytes / 1024, digits=1)) KB each")
println("  Total ≈ $(round(iterations * frame_bytes / 1024^2, digits=1)) MB")
println("  Output : $out_path")
