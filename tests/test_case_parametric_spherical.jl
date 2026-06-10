# =============================================================================
# test_case_parametric_spherical.jl
#
# Great-circle swell propagation test on a spherical (lon/lat) grid.
#
# Physical scenario
# -----------------
# A narrow directional swell system is placed at the equator (lon 180°, lat 0°)
# propagating toward the NE (45° from east).  There is no wind forcing; energy
# is injected purely through the initial condition.  The swell should follow the
# great-circle geodesic, which on a lat-lon map appears as a curved arc bending
# toward higher latitudes.  The SphericalPropagationCorrection (tan(lat)/R term)
# encodes this curvature.
#
# Key diagnostics
# ---------------
# 1. Energy heatmaps over time  → see the swell beam propagate NE then curve.
# 2. Energy centroid trajectory → the centroid path should arc, not be straight.
#
# Grid: lon 0–360°, lat −80–80°, 2° resolution, periodic in longitude.
# =============================================================================

ENV["JULIA_INCREMENTAL_COMPILE"] = true
using Pkg
Pkg.activate(".")
Pkg.add("GeoDatasets")

using PiCLES
using PiCLES.Operators.core_2D_parametric: ParticleDefaultsParam
using PiCLES.Models.ParametricModels: Parametric2D
using PiCLES.Simulations
using PiCLES.Grids.SphericalGrid: TwoDSphericalGridMesh, TwoDSphericalGridStatistics

using PiCLES.ParticleSystems: particle_waves_v7 as PW
using PiCLES.Operators.mapping_2D: reset_PI_u!, ParticleToNode!
using Oceananigans.Units

import Plots as plt
using GeoDatasets
using LinearAlgebra


# =============================================================================
# 1. Parameters
# =============================================================================

DT      = 30minutes
t_final = 30days
r_g0    = 0.85

# First swell characteristics
fp1    = 0.071                    # peak frequency [Hz], period ≈ 14 s
c_g1   = 9.81 / (4π * fp1)        # deep-water group speed ≈ 11 m/s
θ_m1   = -57.5 / 180 * π             # propagation direction: 45° NE from east
β      = 20.0 / 180 * π           # directional half-width [rad] — narrow swell beam
σ_along = 0.75                    # speed spread in the direction along the swell beam [m/s]

σ_accross = 2 * abs(tan(β/2)) * c_g1        # speed spread in the direction across the swell beam [m/s]
σ_cx1   = σ_accross * abs(sin(θ_m1)) + σ_along * abs(cos(θ_m1))   # speed spread in x-velocity space [m/s]
σ_cy1   = σ_accross * abs(cos(θ_m1)) + σ_along * abs(sin(θ_m1))          # speed spread in y-velocity space [m/s]
σ_cxcy  = ((1 - (σ_accross*σ_along) / (σ_cx1*σ_cy1))) * (σ_cx1*σ_cy1)   # covariance between x and y velocity [m²/s²]

# [((σ_cx1+σ_cx2*second_swell))^2  σ_cxcy^2  ; σ_cxcy^2    ((σ_cy1+σ_cy2*second_swell))^2]
# σ_along, σ_accross, σ_cx1, σ_cy1, σ_cxcy, σ_cxcy / (σ_cx1*σ_cy1)

Hs_max1 = 16.0                    # peak significant wave height [m]
E_max1  = Hs_max1^2 / 16          # peak wave energy [m²]


# Second swell characteristics
second_swell = false
fp2    = second_swell*0.071                    # peak frequency [Hz], period ≈ 14 s
c_g2   = second_swell*9.81 / (4π * fp2)        # deep-water group speed ≈ 11 m/s
θ_m2   = second_swell* -0.8*π / 4                   # propagation direction: 45° SE from east
σ_θ2   = second_swell*5.0 * π / 180           # directional half-width [rad] — narrow swell beam
σ_cy2   = second_swell*c_g2 * tan(σ_θ2/2)*2          # speed spread in x-velocity space [m/s]
σ_cx2   = second_swell*c_g2 * 0.2          # speed spread in y-velocity space [m/s]

Hs_max2 = second_swell*4.0                     # peak significant wave height [m]
E_max2  = second_swell*Hs_max2^2 / 16           # peak wave energy [m²]

# First initial swell patch centre (degrees)
lon01   = 160.0
lat01   = 45.0
σ_lon1  = 1.5                    # Gaussian half-width in longitude [°]
σ_lat1  = 1.5                    # Gaussian half-width in latitude  [°]

# Second initial swell patch centre (degrees)
lon02   = 90.0
lat02   = -10.0
σ_lon2  = 2.5                    # Gaussian half-width in longitude [°]
σ_lat2  = 2.5                    # Gaussian half-width in latitude  [°]

# Reference values for MinimalWindsea / MinimalParticle seed (not used as forcing)
U10_ref = 5.0
V10_ref = 5.0

# Grid
lon_min, lon_max, Nlon = 0.0, 360.0, 505
lat_min, lat_max, Nlat = -80.0,  80.0, 225


# =============================================================================
# 2. Zero wind forcing
# =============================================================================

u_zero(lon, lat, t) = 0.0
v_zero(lon, lat, t) = 0.0
winds = (u = u_zero, v = v_zero)


# =============================================================================
# 3. Land mask (GeoDatasets land-sea mask, nearest-neighbour regrid)
# =============================================================================

# lsm: 1 = land, 0 = ocean
lsm_lon, lsm_lat, lsm_data = GeoDatasets.landseamask(resolution='c', grid=5)

lon_grid = collect(range(lon_min, lon_max, length=Nlon))
lat_grid = collect(range(lat_min, lat_max, length=Nlat))

# Wrap simulation longitudes to [-180, 180] to match GeoDatasets convention
function nearest_lsm(lo, la)
    lo_w = mod(lo + 180.0, 360.0) - 180.0
    i = argmin(abs.(lsm_lon .- lo_w))
    j = argmin(abs.(lsm_lat .- la))
    return lsm_data[i, j]
end

# true = ocean, false = land  (PiCLES convention: 1 = ocean, 0 = land)
mask = [nearest_lsm(lo, la) == 0 for lo in lon_grid, la in lat_grid]

@info "Land mask built" sum(mask) "ocean cells out of" length(mask)


# =============================================================================
# 4. Build the spherical grid
# =============================================================================

grid = TwoDSphericalGridMesh(
    lon_min, lon_max, Nlon,
    lat_min, lat_max, Nlat;
    mask              = mask,
    periodic_boundary = (true, false)
)

@info "Grid built" grid.stats.dx_deg grid.stats.dy_deg


# =============================================================================
# 5. ODE system and settings
# =============================================================================

ODEpars, Const_ID, Const_Scg = PW.ODEParameters(r_g = r_g0)

particle_system = PW.particle_equations(
    u_zero, v_zero;
    γ = Const_ID.γ,
    q = Const_ID.q,
    dissipation = false,
    peak_shift = false
    )

WindSeamin = FetchRelations.MinimalWindsea(U10_ref, V10_ref, DT)

# Default covariance (overridden per-node in the IC loop below)
R_earth = 6371.0e3
dx0 = grid.stats.dx_deg * π / 180 * R_earth   # equatorial dx [m]
dy0 = grid.stats.dy_deg * π / 180 * R_earth   # dy [m]
initCovMatrix = [σ_cx1^2  σ_cxcy    0.0    0.0   ;
                 σ_cxcy    σ_cy1^2  0.0    0.0   ;
                 0.0    0.0    dx0^2  0.0   ;
                 0.0    0.0    0.0    dy0^2 ]

default_particle = ParticleDefaultsParam(
    WindSeamin["lne"],
    WindSeamin["cg_bar_x"],
    WindSeamin["cg_bar_y"],
    0.0, 0.0,
    initCovMatrix
)

ODE_settings = PW.ODESettings(
    Parameters         = ODEpars,
    log_energy_minimum = WindSeamin["lne"],
    log_energy_maximum = log(27),
    saving_step        = DT,
    timestep           = DT,
    total_time         = t_final,
    maxiters           = 100000,
    adaptive           = true,
    dt                 = 1e-3,
    dtmin              = 1e-4,
    force_dtmin        = true,
    callbacks          = nothing,
    save_everystep     = false
)


# =============================================================================
# 6. Build the model
# =============================================================================

wave_model = Parametric2D(;
    grid              = grid,
    winds             = winds,
    ODEsys            = particle_system,
    ODEsets           = ODE_settings,
    ODEinit_type      = default_particle,
    periodic_boundary = true,
    boundary_type     = "same",
    minimal_particle  = FetchRelations.MinimalParticle(U10_ref, V10_ref, DT),
    movie             = true
)


# =============================================================================
# 7. Initialise and set IC
# =============================================================================

wave_simulation = Simulation(wave_model, Δt = DT, verbose = true, stop_time = t_final)

initialize_simulation!(wave_simulation)

function gaussian_patch(lon, lat, lon_c, lat_c, σ_lon, σ_lat)
    dlon = lon - lon_c
    dlon = dlon - 360.0 * round(dlon / 360.0)   # longitude wrap
    dlat = lat - lat_c
    return exp(-0.5 * (dlon / σ_lon)^2 - 0.5 * (dlat / σ_lat)^2)
end

function unfold(M::Matrix{Float64})
    return M[1,1], M[1,2], M[2,2], M[1,3], M[2,3], M[1,4], M[2,4], M[3,3], M[3,4], M[4,4]
end

# First swell characteristics

lne_min = ODE_settings.log_energy_minimum
nNonPosDef = 0

for i in 1:Nlon
    for j in 1:Nlat
        PI = wave_simulation.model.ParticleCollection[i, j]
        isnothing(PI.ODEIntegrator) && continue   # skip land / dummy particles

        lon = PI.position_xy[1]   # degrees
        lat = PI.position_xy[2]   # degrees

        g_val1  = gaussian_patch(lon, lat, lon01, lat01, σ_lon1, σ_lat1)
        g_val2  = gaussian_patch(lon, lat, lon02, lat02, σ_lon2, σ_lat2)
        energy = max(exp(lne_min), E_max1 * g_val1 + E_max2 * g_val2)

        # Per-node spatial covariance: one grid cell squared [m²]
        dx_ij = wave_simulation.model.grid.data.dx[i, j]
        dy_ij = wave_simulation.model.grid.data.dy[i, j]

        cov_matrix = [((σ_cx1))^2  σ_cxcy    0.0      0.0    ;
                      σ_cxcy    ((σ_cy1))^2  0.0      0.0    ;
                      0.0    0.0    dx_ij^2  0.0    ;
                      0.0    0.0    0.0      dy_ij^2]
        global nNonPosDef += 1-Int64(isposdef(cov_matrix))

        if E_max1 * g_val1 >= E_max2 * g_val2
            ui = [log(energy),
                  c_g1 * cos(θ_m1),
                  c_g1 * sin(θ_m1),
                  0.0, 0.0,
                  unfold(cov_matrix)...]
        else
            ui = [log(energy),
                  c_g2 * cos(θ_m2),
                  c_g2 * sin(θ_m2),
                  0.0, 0.0,
                  unfold(cov_matrix)...]
        end

        reset_PI_u!(PI, ui = ui)
        ParticleToNode!(PI, [0.0, 0.0],
                        wave_simulation.model.State,
                        wave_simulation.model.grid,
                        wave_simulation.model.periodic_boundary)
    end
end

@info "IC set: swell patch at ($(lon01)°, $(lat01)°), θ=$(round(θ_m1*180/π, digits=0))°, c_g=$(round(c_g1, digits=2)) m/s"
@info "IC set: swell patch at ($(lon02)°, $(lat02)°), θ=$(round(θ_m2*180/π, digits=0))°, c_g=$(round(c_g2, digits=2)) m/s"


run!(wave_simulation, cash_store = true)

@info "Simulation complete."


# =============================================================================
# 8. Post-processing
# =============================================================================

post_process = false

if post_process
    mkpath("plots/test_case_parametric/energy")
    mkpath("plots/test_case_parametric/trajectory")

    lons = grid.data.x[:, 1]   # (Nlon,) longitude vector [°]
    lats = grid.data.y[1, :]   # (Nlat,) latitude vector  [°]

    n_snapshots = length(wave_simulation.store.store)
    time_hours  = collect(1:n_snapshots) .* (DT / 3600.0)

    # Track energy centroid (lon, lat) at each snapshot
    centroid_lon = fill(NaN, n_snapshots)
    centroid_lat = fill(NaN, n_snapshots)

    for i in 1:n_snapshots
        fstate = wave_simulation.store.store[i]
        E      = fstate[:, :, 1]

        # Energy heatmap every 6 snapshots (≈ every 2 hours)
        if mod(i, 6) == 0
            E_plot = copy(E)
            E_plot[wave_model.grid.data.mask .== 0] .= NaN
            plt.heatmap(
                lons, lats, transpose(E_plot),
                title  = "Energy [m²]  t = $(round(time_hours[i], digits=1)) h",
                xlabel = "Longitude (°)",
                ylabel = "Latitude (°)",
                clims  = (0, E_max1),
                size   = (1200, 500)
            )
            plt.savefig("plots/test_case_parametric/energy/energy_$(lpad(i, 4, '0')).png")
        end

        # Energy centroid
        total_E = sum(E)
        if total_E > 1e-8
            centroid_lon[i] = sum(lons  .* E) / total_E
            centroid_lat[i] = sum(lats' .* E) / total_E
        end
    end


    # ---- Energy centroid trajectory --------------------------------------------
    # On a sphere, a NE-propagating swell follows a great circle.
    # On a lat-lon map, this great-circle arc curves northward and then back south.
    # SphericalPropagationCorrection (tan(lat)/R_earth applied to c_x) drives this.

    valid = .!isnan.(centroid_lon)

    plt.scatter(
        centroid_lon[valid], centroid_lat[valid],
        marker_z        = time_hours[valid],
        zcolor          = time_hours[valid],
        title           = "Energy centroid trajectory  (colour = time [h])",
        xlabel          = "Longitude (°)",
        ylabel          = "Latitude (°)",
        colorbar_title  = "Time [h]",
        legend          = false,
        markersize      = 4,
        size            = (1000, 600)
    )
    # Overlay the expected straight-line diagonal for reference
    t_span     = range(0, t_final, length=200)
    gc_lon_ref = @. lon01 + (c_g1 * cos(θ_m1) * t_span) * (180 / π) / R_earth
    gc_lat_ref = @. lat01 + (c_g1 * sin(θ_m1) * t_span) * (180 / π) / R_earth
    plt.plot!(gc_lon_ref, gc_lat_ref, lc = :red, ls = :dash, label = "Flat-earth reference")
    plt.savefig("plots/test_case_parametric/trajectory/centroid_trajectory.png")

    @info "Centroid at final time: lon=$(round(centroid_lon[end], digits=2))°, lat=$(round(centroid_lat[end], digits=2))°"

    # ---- Centroid lon and lat vs time ----------------------------------------
    plt.plot(
        time_hours[valid],
        [centroid_lon[valid]  centroid_lat[valid]],
        label  = ["Centroid longitude" "Centroid latitude"],
        title  = "Energy centroid position over time",
        xlabel = "Time [h]",
        ylabel = "Degrees",
        lw     = 2,
        size   = (900, 500)
    )
    plt.savefig("plots/test_case_parametric/trajectory/centroid_vs_time.png")

    # ---- Final energy snapshot ------------------------------------------------
    E_final = copy(wave_simulation.store.store[end][:, :, 1])
    E_final[wave_model.grid.data.mask .== 0] .= NaN
    plt.heatmap(
        lons, lats, transpose(E_final),
        title  = "Final energy [m²]  t = $(round(time_hours[end], digits=1)) h",
        xlabel = "Longitude (°)",
        ylabel = "Latitude (°)",
        clims  = (0, E_max1),
        size   = (1200, 500)
    )
    plt.savefig("plots/test_case_parametric/trajectory/final_energy.png")

    @info "All plots saved to plots/test_case_parametric/"
    @info "Test case complete."
end