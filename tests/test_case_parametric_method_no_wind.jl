# =============================================================================
# test_case_parametric_method_no_wind.jl
#
# Free swell propagation test on a flat (Cartesian) grid.
#
# Physical scenario
# -----------------
# A narrow directional swell system is placed at the western edge of the domain
# and propagates eastward (θ = 45° NE from east) with no wind forcing.
# Energy is injected purely through the initial condition.
#
# Grid: 0–6600 km × 0–2200 km, ~55 km resolution (121 × 41 nodes).
# =============================================================================

ENV["JULIA_INCREMENTAL_COMPILE"] = true
using Pkg
Pkg.activate(".")

using PiCLES
using PiCLES.Operators.core_2D_parametric: ParticleDefaultsParam
using PiCLES.Models.ParametricModels: Parametric2D
using PiCLES.Simulations
using PiCLES.Grids.CartesianGrid: TwoDCartesianGridMesh
using PiCLES.ParticleSystems: particle_waves_v7 as PW
using PiCLES.Operators.mapping_2D: reset_PI_u!, ParticleToNode!
using Oceananigans.Units

import Plots as plt


# =============================================================================
# 1. Parameters
# =============================================================================

remesh_kern = "CIC"
output_dir = "plots/test_case_parametric/data"
DT      = 1hours
t_final = 2hours
r_g0    = 0.85

# Swell characteristics
fp      = 0.071                         # peak frequency [Hz], period ≈ 14 s
c_g     = 9.81 / (4π * fp)              # deep-water group speed ≈ 11 m/s
θ_m     = 0.0 / 180 * π                # propagation direction [rad] from east (45° NE)
β       = 40.0 / 180 * π                # directional half-width [rad]
σ_along = 0.75                          # speed spread along swell direction [m/s]

σ_across = 2 * abs(tan(β / 2)) * c_g
σ_cx     = σ_across * abs(sin(θ_m)) + σ_along * abs(cos(θ_m))
σ_cy     = σ_across * abs(cos(θ_m)) + σ_along * abs(sin(θ_m))
σ_cxcy   = -(1 - σ_across * σ_along / (σ_cx * σ_cy)) * σ_cx * σ_cy

Hs_max  = 16.0                           # peak significant wave height [m]
E_max   = Hs_max^2 / 16                 # peak wave energy [m²]

# Initial swell patch centre and spatial spread
x0  = 400e3                               # patch centre x [m]
y0  = 3300e3                            # patch centre y [m]
σ_x = 25e3                             # spatial spread in x [m]
σ_y = 25e3                             # spatial spread in y [m]

# Grid
xmax, Nx = 6600e3, 121
ymax, Ny = 6600e3, 121

U10_ref = 10.0
V10_ref = 10.0


# =============================================================================
# 2. Zero wind forcing
# =============================================================================

u_zero(x, y, t) = 0.0
v_zero(x, y, t) = 0.0
winds = (u = u_zero, v = v_zero)


# =============================================================================
# 3. Build the Cartesian grid
# =============================================================================

grid = TwoDCartesianGridMesh(xmax, Nx, ymax, Ny)

@info "Grid built" grid.stats.dx grid.stats.dy


# =============================================================================
# 4. ODE system and settings
# =============================================================================

ODEpars, Const_ID, Const_Scg = PW.ODEParameters(r_g = r_g0)

particle_system = PW.particle_equations(u_zero, v_zero;
    γ           = Const_ID.γ,
    q           = Const_ID.q,
    peak_shift  = true,
    dissipation = false
)

WindSeamin = FetchRelations.MinimalWindsea(U10_ref, V10_ref, DT)

initCovMatrix = [σ_cx^2   σ_cxcy   0.0              0.0             ;
                 σ_cxcy   σ_cy^2   0.0              0.0             ;
                 0.0      0.0      grid.stats.dx^2  0.0             ;
                 0.0      0.0      0.0              grid.stats.dy^2 ]

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
    saving_step        = DT,
    timestep           = DT,
    total_time         = t_final,
    dt                 = 1e-3,
    dtmin              = 1e-4,
    force_dtmin        = true
)


# =============================================================================
# 5. Build the model
# =============================================================================

wave_model = Parametric2D(;
    grid              = grid,
    winds             = winds,
    ODEsys            = particle_system,
    ODEsets           = ODE_settings,
    ODEinit_type      = default_particle,
    periodic_boundary = false,
    minimal_particle  = FetchRelations.MinimalParticle(U10_ref, V10_ref, DT),
    movie             = true,
    remeshing_kernel  = remesh_kern,
    plot_savepath     = output_dir
)


# =============================================================================
# 6. Initialise and set IC
# =============================================================================

wave_simulation = Simulation(wave_model, Δt = DT, verbose = true, stop_time = t_final)

initialize_simulation!(wave_simulation)

function unfold(M::Matrix{Float64})
    return M[1,1], M[1,2], M[2,2], M[1,3], M[2,3], M[1,4], M[2,4], M[3,3], M[3,4], M[4,4]
end

lne_min = ODE_settings.log_energy_minimum

for i in 1:Nx
    for j in 1:Ny
        PI  = wave_simulation.model.ParticleCollection[i, j]
        x   = PI.position_xy[1]
        y   = PI.position_xy[2]

        g_val  = exp(-0.5 * ((x - x0)^2 / σ_x^2 + (y - y0)^2 / σ_y^2))
        energy = max(exp(lne_min), E_max * g_val)

        cov_matrix = [σ_cx^2   σ_cxcy   0.0              0.0             ;
                      σ_cxcy   σ_cy^2   0.0              0.0             ;
                      0.0      0.0      grid.stats.dx^2  0.0             ;
                      0.0      0.0      0.0              grid.stats.dy^2 ]

        ui = [log(energy),
              c_g * cos(θ_m),
              c_g * sin(θ_m),
              0.0, 0.0,
              unfold(cov_matrix)...]

        reset_PI_u!(PI, ui = ui)
        ParticleToNode!(PI, [0.0, 0.0],
                        wave_simulation.model.State,
                        wave_simulation.model.grid,
                        wave_simulation.model.periodic_boundary, wave_simulation.model.remeshing_kernel)
    end
end

@info "IC set: swell patch at ($(x0/1e3) km, $(y0/1e3) km), θ=$(round(θ_m*180/π, digits=1))°, c_g=$(round(c_g, digits=2)) m/s"


# =============================================================================
# 7. Run
# =============================================================================

run!(wave_simulation, cash_store = true, save_format = ("binary","csv"))

@info "Simulation complete."
