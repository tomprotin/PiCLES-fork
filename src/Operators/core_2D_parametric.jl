module core_2D_parametric

using DifferentialEquations: OrdinaryDiffEq.ODEIntegrator, OrdinaryDiffEq.ODEProblem, init

using SharedArrays
using StaticArrays

using DocStringExtensions

# Particle-Node interaction
export GetParticleEnergyMomentum,GetParticleEnergyMomentumSwell, GetVariablesAtVertex, Get_u_FromShared, ParticleDefaultsParam, InitParticleInstance
export InitParticleValues

#include("../Utils/FetchRelations.jl")
using ...FetchRelations

using ...Architectures: AbstractParticleInstance, AbstractMarkedParticleInstance, StateTypeL1
using ...Architectures: AbstractGridStatistics, CartesianGridStatistics, SphericalGridStatistics, TripolarGridStatistics

# -----------------------------------------------------------------------
# get_mesh_size(gridstats, ij_mesh) → Vector{Float64}([dx_metres, dy_metres])
#
# Returns the physical cell dimensions in metres for the current grid node.
#
# Dispatch logic:
#   - CartesianGridStatistics: dx and dy are uniform scalars stored in the
#     stats struct itself (metres).  ij_mesh does not carry dx/dy fields.
#   - SphericalGridStatistics / TripolarGridStatistics: dx and dy vary with
#     latitude and are stored per-node in the StructArray.  The ij_mesh
#     NamedTuple (a row of grid.data) already carries ij_mesh.dx and
#     ij_mesh.dy in metres.
#
# Adding support for a new grid type requires only adding a new method here.
# -----------------------------------------------------------------------

"""
    get_mesh_size(gridstats::CartesianGridStatistics, ij_mesh::NamedTuple) → Vector{Float64}

Return the uniform Cartesian cell dimensions [dx, dy] in metres from the
grid statistics struct.
"""
function get_mesh_size(gridstats::CartesianGridStatistics, ij_mesh::NamedTuple)
    return [gridstats.dx, gridstats.dy]
end

"""
    get_mesh_size(gridstats::SphericalGridStatistics, ij_mesh::NamedTuple) → Vector{Float64}

Return the local cell dimensions [dx, dy] in metres from the per-node data
stored in the spherical grid StructArray row.  On a spherical grid dx varies
with latitude (dx ∝ cos(lat)), so the uniform degree-based value in gridstats
cannot be used directly.
"""
function get_mesh_size(gridstats::SphericalGridStatistics, ij_mesh::NamedTuple)
    return [ij_mesh.dx, ij_mesh.dy]
end

"""
    get_mesh_size(gridstats::TripolarGridStatistics, ij_mesh::NamedTuple) → Vector{Float64}

Return the local cell dimensions [dx, dy] in metres from the per-node data
stored in the tripolar grid StructArray row.
"""
function get_mesh_size(gridstats::TripolarGridStatistics, ij_mesh::NamedTuple)
    return [ij_mesh.dx, ij_mesh.dy]
end


# using ..particle_waves_v3: init_vars
# t, x, y, c̄_x, c̄_y, lne, Δn, Δφ_p, r_g, C_α, C_φ, g, C_e = init_vars()

using ...custom_structures: ParticleInstance1D, ParametricParticleInstance2D, MarkedParticleInstance

export init_z0_to_State!
include("initialize.jl")

# Callbacks
export wrap_pos!, periodic_BD_single_PI!, show_pos!, periodic_condition_x
include("utils.jl")

function fold(v::Vector{Float64})
        return [v[1] v[2] v[4] v[6]; v[2] v[3] v[5] v[7]; v[4] v[5] v[8] v[9]; v[6] v[7] v[9] v[10]]
end

function unfold(M::Matrix{Float64})
        return M[1,1], M[1,2], M[2,2], M[1,3], M[2,3], M[1,4], M[2,4], M[3,3], M[3,4], M[4,4]
end

### particle defaults ###
"""
ParticleDefaultsParam
Structure holds default particles
# Fields  
$(DocStringExtensions.FIELDS)
"""
mutable struct ParticleDefaultsParam{T<:AbstractFloat}
        "log energy"
        lne::T
        "zonal velocity"
        c̄_x::T
        "meridional velocity"
        c̄_y::T
        "x Position"
        x::T
        "y Position"
        y::T
        "X K covariance"
        cov_xk::Matrix{T}
end

ParticleDefaultsParam(vec::Vector{Float64}) = ParticleDefaultsParam(vec[1], vec[2], vec[3], vec[4], vec[5], fold(vec[6:end]))

Base.copy(s::ParticleDefaultsParam) = ParticleDefaultsParam(s.lne, s.c̄_x, s.c̄_y, s.x, s.y, s.cov_xk)
#initParticleDefaults(s::ParticleDefaultsParam) = MVector{5,Float64}([s.lne, s.c̄_x, s.c̄_y, s.x, s.y])
#initParticleDefaults(s::ParticleDefaultsParam) = MVector{5,Float64}(s.lne, s.c̄_x, s.c̄_y, s.x, s.y)
initParticleDefaults(s::ParticleDefaultsParam) = [s.lne, s.c̄_x, s.c̄_y, s.x, s.y, unfold(s.cov_xk)...]




######### Particle --> Node  |  2D  ##########

"""
GetParticleEnergyMomentum(PI)

"""
function GetParticleEnergyMomentum(z0::TT, mesh_size::Vector{Float64}) where {TT<:Union{Vector{Float64},MVector{15,Float64},ParticleDefaultsParam}}

        if z0 isa Vector{Float64} || z0 isa MVector{15,Float64}
                ui_lne, ui_c̄_x, ui_c̄_y, _, _, ui_xx, ui_xy, ui_yy, ui_xkx, ui_ykx, ui_xky, ui_yky, ui_kxkx, ui_kxky, ui_kyky = z0
        elseif z0 isa ParticleDefaultsParam
                ui_lne, ui_c̄_x, ui_c̄_y, _, _, _ = z0.lne, z0.c̄_x, z0.c̄_y, z0.x, z0.y, z0.cov_xk
        else
                error("input should be either Vector{Float64}, MVector{15,Float64} or ParticleDefaultsParam")
        end

        ui_e = exp(ui_lne)
        c_speed = speed(ui_c̄_x, ui_c̄_y)
        m_x = ui_c̄_x * ui_e / c_speed^2 / 2
        m_y = ui_c̄_y * ui_e / c_speed^2 / 2

        ui_M = [(1/sqrt(2)*c_speed/2)^2 -(1/sqrt(2)*c_speed/2)^2*0.5 0 0;
                -(1/sqrt(2)*c_speed/2)^2*0.5 (1/sqrt(2)*c_speed/2)^2 0 0;            # TO BE CHANGED
                0 0 mesh_size[1]^2 0;
                0 0 0 mesh_size[2]^2
        ]
        m_Mxk = ui_M * ui_e / c_speed^2 / 2

        return SVector{13,Float64}(ui_e, m_x, m_y, unfold(m_Mxk)...)
end

function local_mean_speed_gaussian_2D(X, X0, cov)
        res = X0[1:2] + cov[1:2, 3:4] * inv(cov[3:4, 3:4]) * (X - X0[3:4])
        return res
end

function local_cov_speed_gaussian_2D(X, X0, cov)
        res = cov[1:2, 1:2] - cov[1:2, 3:4] * inv(cov[3:4, 3:4]) * cov[3:4, 1:2]
        return res
end

function GetParticleEnergyMomentumSwell(z0::TT, mesh_size::Vector{Float64}, remeshing_targets::Matrix{Int64}) where {TT<:Union{Vector{Float64},MVector{15,Float64},ParticleDefaultsParam}}

        if z0 isa Vector{Float64} || z0 isa MVector{15,Float64}
                ui_lne, ui_c̄_x, ui_c̄_y, x, y, ui_xx, ui_xy, ui_yy, ui_xkx, ui_ykx, ui_xky, ui_yky, ui_kxkx, ui_kxky, ui_kyky = z0
                ui_M = fold([ui_xx, ui_xy, ui_yy, ui_xkx, ui_ykx, ui_xky, ui_yky, ui_kxkx, ui_kxky, ui_kyky])
        elseif z0 isa ParticleDefaultsParam
                ui_lne, ui_c̄_x, ui_c̄_y, x, y, ui_M = z0.lne, z0.c̄_x, z0.c̄_y, z0.x, z0.y, z0.cov_xk
        else
                error("input should be either Vector{Float64}, MVector{15,Float64} or ParticleDefaultsParam")
        end

        nRemeshingTargets = size(remeshing_targets, 1)

        dx = mesh_size[1]
        dy = mesh_size[2]
        x = (x - floor(x)) * dx
        y = (y - floor(y)) * dy
        # x = x*dx
        # y = y*dy

        uis = [local_mean_speed_gaussian_2D([remeshing_targets[i,1]*dx, remeshing_targets[i,2]*dy], [ui_c̄_x, ui_c̄_y, x, y], ui_M) for i in 1:nRemeshingTargets]

        # ui_c_1 = local_mean_speed_gaussian_2D([0, 0], [ui_c̄_x, ui_c̄_y, x, y], ui_M)
        # ui_c_2 = local_mean_speed_gaussian_2D([dx, 0], [ui_c̄_x, ui_c̄_y, x, y], ui_M)
        # ui_c_3 = local_mean_speed_gaussian_2D([0, dy], [ui_c̄_x, ui_c̄_y, x, y], ui_M)
        # ui_c_4 = local_mean_speed_gaussian_2D([dx, dy], [ui_c̄_x, ui_c̄_y, x, y], ui_M)

        # @info "dx: $dx, dy: $dy"
        # @info "x: $x, y: $y"
        # @info "ui_c̄_x: $ui_c̄_x, ui_c̄_y: $ui_c̄_y"
        # @info "ui_M: $ui_M"
        # @info "ui_c_1: $ui_c_1"
        # @info "ui_c_2: $ui_c_2"
        # @info "ui_c_3: $ui_c_3"
        # @info "ui_c_4: $ui_c_4"
        # @info ""

        c_speeds = [speed(uis[i]...) for i in 1:nRemeshingTargets]

        # c1_speed = speed(ui_c_1[1], ui_c_1[2])
        # c2_speed = speed(ui_c_2[1], ui_c_2[2])
        # c3_speed = speed(ui_c_3[1], ui_c_3[2])
        # c4_speed = speed(ui_c_4[1], ui_c_4[2])

        ui_e = exp(ui_lne)
        ms = [uis[i] .* ui_e / c_speeds[i]^2 / 2 for i in 1:nRemeshingTargets]

        # m_x1 = ui_c_1[1] * ui_e / c1_speed^2 / 2
        # m_x2 = ui_c_2[1] * ui_e / c2_speed^2 / 2
        # m_x3 = ui_c_3[1] * ui_e / c3_speed^2 / 2
        # m_x4 = ui_c_4[1] * ui_e / c4_speed^2 / 2
        # m_y1 = ui_c_1[2] * ui_e / c1_speed^2 / 2
        # m_y2 = ui_c_2[2] * ui_e / c2_speed^2 / 2
        # m_y3 = ui_c_3[2] * ui_e / c3_speed^2 / 2
        # m_y4 = ui_c_4[2] * ui_e / c4_speed^2 / 2

        ui_Ms = [local_cov_speed_gaussian_2D([remeshing_targets[i,1]*dx, remeshing_targets[i,2]*dy], [ui_c̄_x, ui_c̄_y, x, y], ui_M) for i in 1:nRemeshingTargets]

        # ui_M_C1 = local_cov_speed_gaussian_2D([0, 0], [ui_c̄_x, ui_c̄_y, x, y], ui_M)
        # ui_M_C2 = local_cov_speed_gaussian_2D([1, 0], [ui_c̄_x, ui_c̄_y, x, y], ui_M)
        # ui_M_C3 = local_cov_speed_gaussian_2D([0, 1], [ui_c̄_x, ui_c̄_y, x, y], ui_M)
        # ui_M_C4 = local_cov_speed_gaussian_2D([1, 1], [ui_c̄_x, ui_c̄_y, x, y], ui_M)

        final_ui_Ms = [[ui_Ms[i] [0 0;0 0]; [0 0;0 0] [1 0; 0 1]] for i in 1:nRemeshingTargets]

        # final_ui_M1 = [ui_M_C1[1,1] ui_M_C1[1,2] 0 0;
        #               ui_M_C1[2,1] ui_M_C1[2,2] 0 0;
        #               0 0 1 0;
        #               0 0 0 1
        # ]
        # final_ui_M2 = [ui_M_C2[1,1] ui_M_C2[1,2] 0 0;
        #               ui_M_C2[2,1] ui_M_C2[2,2] 0 0;
        #               0 0 1 0;
        #               0 0 0 1
        # ]
        # final_ui_M3 = [ui_M_C3[1,1] ui_M_C3[1,2] 0 0;
        #               ui_M_C3[2,1] ui_M_C3[2,2] 0 0;
        #               0 0 1 0;
        #               0 0 0 1
        # ]
        # final_ui_M4 = [ui_M_C4[1,1] ui_M_C4[1,2] 0 0;
        #               ui_M_C4[2,1] ui_M_C4[2,2] 0 0;
        #               0 0 1 0;
        #               0 0 0 1
        # ]

        m_Ms = [final_ui_Ms[i] * ui_e / c_speeds[i]^2 / 2 for i in 1:nRemeshingTargets]

        # m_Mxk1 = final_ui_M1 * ui_e / c1_speed^2 / 2
        # m_Mxk2 = final_ui_M2 * ui_e / c2_speed^2 / 2
        # m_Mxk3 = final_ui_M3 * ui_e / c3_speed^2 / 2
        # m_Mxk4 = final_ui_M4 * ui_e / c4_speed^2 / 2

        return ui_e, ms..., m_Ms...
end


# Aliases: 

# function GetParticleEnergyMomentum(PI)
#         # ui_lne, ui_c̄_x, ui_c̄_y, ui_x, ui_y = PI.ODEIntegrator.u
#         return GetParticleEnergyMomentum(PI.ODEIntegrator.u)
#         # ui_e = exp(ui_lne)
#         # c_speed = speed(ui_c̄_x, ui_c̄_y)
#         # m_x = ui_c̄_x * ui_e / c_speed^2 / 2
#         # m_y = ui_c̄_y * ui_e / c_speed^2 / 2

#         # return SVector{3,Float64}(ui_e, m_x, m_y)
# end

"""
GetParticleEnergyMomentum(PI)

"""
function GetParticleEnergyMomentum(PI::AbstractParticleInstance, mesh_size::Vector{Float64})
        return GetParticleEnergyMomentum(PI.ODEIntegrator.u, mesh_size)
end

function GetParticleEnergyMomentum(zi::Dict, mesh_size::Vector{Float64})
        return GetParticleEnergyMomentum([zi[lne], zi[c̄_x], zi[c̄_y], zi[x], zi[y], zi.cov_xk[1,1], zi.cov_xk[1,2], zi.cov_xk[2,2], zi.cov_xk[1,3], zi.cov_xk[2,3], zi.cov_xk[1,4], zi.cov_xk[2,4], zi.cov_xk[3,3], zi.cov_xk[3,4], zi.cov_xk[4,4]], mesh_size)
end

function GetParticleEnergyMomentum(zi::ParticleDefaultsParam, mesh_size::Vector{Float64})
        return GetParticleEnergyMomentum([zi.lne, zi.c̄_x, zi.c̄_y, zi.x, zi.y, zi.cov_xk[1,1], zi.cov_xk[1,2], zi.cov_xk[2,2], zi.cov_xk[1,3], zi.cov_xk[2,3], zi.cov_xk[1,4], zi.cov_xk[2,4], zi.cov_xk[3,3], zi.cov_xk[3,4], zi.cov_xk[4,4]], mesh_size)
end


speed(x::Float64, y::Float64) = sqrt(x^2 + y^2)

######### node--> Particle ##########

"""
GetVariablesAtVertex(i_State::Vector{Float64}, x::Float64, y::Float64, Δn::Float64, Δφ_p::Float64 )
i_state: [e, m_x, m_y] state vector at node
x, y: coordinates of the vertex

"""
function GetVariablesAtVertex(i_State::TT, x::Float64, y::Float64) where {TT<:Union{Vector{Float64},MVector{4,Float64}}}
    e, m_x, m_y, m_Mxk... = i_State
    m_Mxk = fold(m_Mxk)
    m_amp = speed(m_x, m_y)
    c_x = m_x * e / (2 * m_amp^2)
    c_y = m_y * e / (2 * m_amp^2)
    M_xk = m_Mxk * e / (2 * m_amp^2)

    return MVector{15,Float64}(log(e), c_x, c_y, x, y, unfold(M_xk)...)
end


""" returns node state from shared array, given paritcle index """
Get_u_FromShared(PI::AbstractParticleInstance, S::StateTypeL1,) = S[PI.position_ij[1], PI.position_ij[2], :]

"""
GetGroupVelocity(i_State::Vector{Float64})
i_state: [e, m_x, m_y] state vector at node
"""
function GetGroupVelocity(i_State)
        e = i_State[:,:,1]
        m_x = i_State[:,:,2]
        m_y = i_State[:,:,3]
        m_amp = speed.(m_x, m_y)
        c_x = m_x .* e ./ (2 .* m_amp.^2)
        c_y = m_y .* e ./ (2 .* m_amp.^2)

        return (c_x=c_x, c_y=c_y)
end


###### seed particles #####
"""
InitParticleInstance(model::WaveGrowth1D, z_initials, pars, ij, boundary_flag, particle_on ; cbSets=nothing)
wrapper function to initalize a particle instance
        inputs:
        model           is an initlized ODESytem
        z_initials      is the initial state of the ODESystem
        pars            are the parameters of the ODESystem
        ij              is the (i,j) tuple that of the particle position
        xy              is the (x,y) tuple that of the particle position
        boundary_flag   is a boolean that indicates if the particle is on the boundary
        particle_on     is a boolean that indicates if the particle is on
        chSet           (optional) is the set of callbacks the ODE can have
"""
function InitParticleInstance(model, z_initials, ODE_settings, ij, xy , boundary_flag, particle_on; cbSets=Nothing)

        ## to do's for add the Projection:
        ## replace boundary_flag with mask that discrimiated by types [0,1,2,3]
        ## add M as input to the function and add it to the ODE_parameters
        ## modify ParticleInstance2D to deal with boundaries and mask

        # converty to ordered named tuple
        ODE_parameters = NamedTuple{Tuple(Symbol.(keys(ODE_settings.Parameters)))}(values(ODE_settings.Parameters))
        ODE_parameters = (; ODE_parameters..., x = xy[1], y = xy[2])

        z_initials = initParticleDefaults(z_initials)
        # create ODEProblem
        problem = ODEProblem(model, z_initials, (0.0, ODE_settings.total_time), ODE_parameters)
        # inialize problem
        # works best with abstol = 1e-4,reltol=1e-3,maxiters=1e4,
        integrator = init(
                problem,
                ODE_settings.solver,
                saveat=ODE_settings.saving_step,
                abstol=ODE_settings.abstol,
                adaptive=ODE_settings.adaptive,
                dt = ODE_settings.dt,
                dtmin=ODE_settings.dtmin,
                force_dtmin=ODE_settings.force_dtmin,
                maxiters=ODE_settings.maxiters,
                reltol=ODE_settings.reltol,
                callback=ODE_settings.callbacks,
                save_everystep=ODE_settings.save_everystep)

        return ParametricParticleInstance2D(ij, (xy[1], xy[2]), integrator, boundary_flag, particle_on)
end

# deprechiated because of MTK
# """
# InitParticleInstance(model::ODESystem, z_initials, pars, ij, boundary_flag, particle_on ; cbSets=nothing)
# wrapper function to initalize a particle instance
#         inputs:
#         model           is an initlized ODESytem
#         z_initials      is the initial state of the ODESystem
#         pars            are the parameters of the ODESystem
#         ij              is the (i,j) tuple that of the initial position
#         boundary_flag   is a boolean that indicates if the particle is on the boundary
#         particle_on     is a boolean that indicates if the particle is on
#         chSet           (optional) is the set of callbacks the ODE can have
# """
# function InitParticleInstance(model::ODESystem, z_initials, ODE_settings, ij, boundary_flag, particle_on; cbSets=Nothing)

#         z_initials = initParticleDefaults(z_initials)
#         # create ODEProblem
#         problem = ODEProblem(model, z_initials, (0.0, ODE_settings.total_time), ODE_settings.Parameters)
#         # inialize problem
#         # works best with abstol = 1e-4,reltol=1e-3,maxiters=1e4,
#         integrator = init(
#                 problem,
#                 ODE_settings.solver,
#                 saveat=ODE_settings.saving_step,
#                 abstol=ODE_settings.abstol,
#                 adaptive=ODE_settings.adaptive,
#                 dt=ODE_settings.dt,
#                 dtmin=ODE_settings.dtmin,
#                 force_dtmin=ODE_settings.force_dtmin,
#                 maxiters=ODE_settings.maxiters,
#                 reltol=ODE_settings.reltol,
#                 callback=ODE_settings.callbacks,
#                 save_everystep=ODE_settings.save_everystep)
#         return ParticleInstance2D(ij, (z_initials[4], z_initials[5]), integrator, boundary_flag, particle_on)
# end




"""
InitParticleValues(defaults:: Union{Nothing,ParticleDefaultsParam}, ij::Tuple, gridnote::OneDGridNotes, u, v, DT)

Find initial conditions for particle. Used at the beginning of the experiment.
        inputs:
        defaults        Dict with variables of the state vector, these will be replaced in this function
        ij              index of the grid point
        gridnote        grid to determine the position in the grid
        winds           NamedTuple (u,v) with interp. functions for wind values
        DT              time step of model, used to determine fetch laws
"""
function InitParticleValues(
        defaults::PP,
        mesh_size::Vector{Float64},
        xy::Tuple{Float64, Float64},
        uv::Tuple{Number, Number}, 
        DT) where {PP<:Union{Nothing,ParticleDefaultsParam}}

        #i,j = ij
        xx, yy = xy

        if defaults == nothing

                if speed(uv[1], uv[2]) > sqrt(2)
                        # defaults are not defined and there is wind

                        # take in local wind velocities
                        ui = FetchRelations.get_initial_windsea(uv[1], uv[2], DT; particle_state= true)
                        # seed particle given fetch relations
                        # lne = log(WindSeaMin["E"])
                        # c̄_x = WindSeaMin["cg_bar_x"]
                        # c̄_y = WindSeaMin["cg_bar_y"]

                        particle_on = true
                else
                        # defaults are not defined and there is no wind
                        ui = FetchRelations.MinimalParticle(uv[1], uv[2], DT)
                        #lne = u_min[1]
                        #c̄_x = u_min[2]
                        #c̄_y = u_min[3]
                        
                        particle_on = false
                end        
                
                # initialize particle instance based on above devfined values
                c_speed = speed(ui[2], ui[3])
                particle_defaults = ParticleDefaultsParam(ui[1], ui[2], ui[3] ,  xx, yy, [(1/sqrt(2)*c_speed/2)^2 -(1/sqrt(2)*c_speed/2)^2*0.5 0. 0.; -(1/sqrt(2)*c_speed/2)^2*0.5 (1/sqrt(2)*c_speed/2)^2 0. 0.; 0. 0. (mesh_size[1])^2 0.; 0. 0. 0. (mesh_size[2])^2])            # TO BE CHANGED
        else
                particle_defaults = defaults
                particle_on = true
        end

        #@show defaults
        return particle_defaults, particle_on
end


# --------------------------- up to here for now (remove MTK)--------------------------- #

"""
ResetParticleValues(PI::AbstractParticleInstance, particle_defaults::Union{Nothing,ParticleDefaultsParam,Vector{Float64}}, ODE_settings::ODESettings, gridnote::OneDGridNotes, winds, DT)

Reset the state of a particle instance
        inputs:
        PI                      ParticleInstance
        particle_defaults       Dict with variables of the state vector, these will be replaced in this function
        ODE_settings            ODESettings type
        gridnote                grid to determine the position in the grid
        winds                   NamedTuple (u,v) with interp. functions for wind values
        DT                      time step of model, used to determine fetch laws
returns:
        dict or vector
"""
function ResetParticleValues(
        defaults::PP,
        mesh_size::Vector{Float64},
        xy::Tuple{Float64,Float64}, #PI::AbstractParticleInstance, #< -------- this should be just xy tuple that can be (0,0)
        wind_tuple,
        DT, vector=true) where {PP<:Union{Nothing,ParticleDefaultsParam,Vector{Float64}}}

        if isnothing(defaults) # this is boundary_defaults = "wind_sea"
                #@info "init particles from fetch relations: $z_i"
                #particle_defaults = Dict{Num,Float64}()
                #xx, yy = PI.position_xy[1], PI.position_xy[2]
                # take in local wind velocities
                u_init, v_init = wind_tuple[1], wind_tuple[2] 
                #winds.u(xx, yy, 0), winds.v(xx, yy, 0)

                ui = FetchRelations.get_initial_windsea(u_init, v_init, DT, particle_state=true)
                # seed particle given fetch relations
                c_speed = speed(ui[2], ui[3])
                particle_defaults = ParticleDefaultsParam(ui[1], ui[2], ui[3], xy[1], xy[2], [(1/sqrt(2)*c_speed/2)^2 -(1/sqrt(2)*c_speed/2)^2*0.5 0. 0.; -(1/sqrt(2)*c_speed/2)^2*0.5 (1/sqrt(2)*c_speed/2)^2 0. 0.; 0. 0. (mesh_size[1])^2 0.; 0. 0. 0. (mesh_size[2])^2])

        elseif typeof(defaults) == Vector{Float64} # this is for the case of minimal wind sea
                particle_defaults = defaults
                particle_defaults[4]    = xy[1]
                particle_defaults[5]    = xy[2]
        else
                # particle_defaults = defaults
                # particle_defaults[4] = xy[1]
                # particle_defaults[5] = xy[2]
                particle_defaults = ParticleDefaultsParam(defaults.lne, defaults.c̄_x, defaults.c̄_y, xy[1], xy[2], defaults.cov_xk)
        end

        #@show defaults
        if vector
                return initParticleDefaults(particle_defaults)
        else
                return particle_defaults
        end

end
"""

check_boundary_point(i, boundary, periodic_boundary)
checks where ever or not point is a boundary point.
## depreciate this version
returns Bool
"""
function check_boundary_point(i, boundary, periodic_boundary)
        return periodic_boundary ? false : (i in boundary)
end

"""
check_boundary_point(imesh, periodic_boundary)
checks if point is boundary point
        returns Bool
"""
function check_boundary_point(imesh, periodic_boundary)
        if periodic_boundary
                return imesh == 2 # only land points are treated as boundary points
        else
                return imesh >= 2 # land points and grid boundary points are treated as boundary points
        end
end

# """
# SeedParticle!(ParticleCollection ::Vector{Any}, State::SharedMatrix, i::Int64,
#                 particle_system::ODESystem, particle_defaults::Union{ParticleDefaultsParam,Nothing}, ODE_settings,
#                 GridNotes, winds, DT:: Float64, Nx:: Int, boundary::Vector{Int}, periodic_boundary::Bool)

# Seed Pickles to ParticleColletion and State
# """
# function SeedParticle!(
#         ParticleCollection::Vector{Any},
#         State::SharedArray,
#         ij::Tuple{Int, Int},

#         particle_system::SS,
#         particle_defaults::PP,
#         ODE_settings, #particle_waves_v3.ODESettings type

#         GridNotes, # ad type of grid note
#         winds,     # interp winds
#         DT::Float64,

#         boundary::Vector{T},
#         periodic_boundary::Bool) where {T<:Union{Int,Any,Nothing,Int64},PP<:Union{ParticleDefaultsParam,Nothing},SS<:Union{ODESystem,Any}}

#         xx, yy = GridNotes.x[ij[1]], GridNotes.y[ij[2]]

        
#         #uv = winds.u(xx, yy, 0)::Union{Num,Float64}, winds.v(xx, yy, 0)::Union{Num,Float64}
#         uv = winds.u(xx, yy, 0)::Float64, winds.v(xx, yy, 0)::Float64

#         # define initial condition
#         z_i, particle_on = InitParticleValues(particle_defaults, (xx,yy), uv, DT)
#         # check if point is boundary point
#         boundary_point = check_boundary_point(ij, boundary, periodic_boundary)
#         #@info "boundary?", boundary_point

#         # if boundary_point
#         #     #@info "boundary point", boundary_point, particle_on
#         #     # if boundary point, then set particle to off
#         #     particle_on = false
#         # end

#         # add initial state to State vector
#         if particle_on
#                 init_z0_to_State!(State, ij, GetParticleEnergyMomentum(z_i))
#         end

#         # Push Inital condition to collection
#         push!(ParticleCollection,
#         InitParticleInstance(
#                 particle_system,
#                 z_i,
#                 ODE_settings,
#                 ij,
#                 boundary_point,
#                 particle_on))
#         nothing
# end


"""
        function SeedParticle(State::StateTypeL1, ij:: (Int64, Int64)
        particle_system::Any, particle_defaults::Union{ParticleDefaultsParam,Nothing}, ODE_settings,
        ij_mesh, ij_wind_tuple, DT:: Float64, boundary::Vector{Int}, periodic_boundary::Bool)

        returns ParicleInstance that can be pushed to ParticleColletion
"""
function SeedParticle(
        State::StateTypeL1,
        ij::II, # position tuple in grid
        
        particle_system::SS,
        particle_defaults::PP,
        ODE_settings, #particle_waves_v3.ODESettings type
        
        gridstats::AbstractGridStatistics,
        ProjetionKernel::Function,
        PropagationCorrection::Function,

        ij_mesh::NamedTuple, # local grid information
        ij_wind::Tuple,     # interp winds
        
        DT::Float64,
        boundary::Vector{T}, periodic_boundary::Bool) where 
        {II<:Union{Tuple{Int,Int},CartesianIndex},T<:Union{Int,Any,Nothing,Int64},PP<:Union{ParticleDefaultsParam,Nothing},SS<:Any}

        xy = (ij_mesh.x, ij_mesh.y)
        # 1st check if particle is not in mask, Land points == 0
        if ij_mesh.mask == 0 # land point
                # init dummy instance
                return ParametricParticleInstance2D(ij, xy , nothing, false, false)
        end

        # define initial condition
        # particle initial condition is always (0,0) in relative coordinates not xy anymore
        mesh_size = get_mesh_size(gridstats, ij_mesh)
        z_i, particle_on = InitParticleValues(particle_defaults, mesh_size, (0.0, 0.0) , ij_wind, DT)

        # check if point is boundary point <-- replace in the future with with mask: 0 = land, 1 = ocean, 2= land boundary, 3 = domain boundary
        # boundary_point = check_boundary_point(ij, boundary, periodic_boundary) # old version that compares to list 
        boundary_point   = check_boundary_point(ij_mesh.mask, periodic_boundary)

        # add initial state to State vector
        if particle_on
                init_z0_to_State!(State, ij, GetParticleEnergyMomentum(z_i, mesh_size))
        end

        # check if Propgation Correction is set in gridstats
        # PropagationCorrection   = gridstats.PropagationCorrection != nothing ? PropagationCorrection(ij_mesh, gridstats) : x -> 0.0

        # set projection:
        ODE_settings.Parameters = (; ODE_settings.Parameters..., M=ProjetionKernel(ij_mesh, gridstats), PC=PropagationCorrection(ij_mesh, gridstats))

        return InitParticleInstance(
                particle_system,
                z_i,
                ODE_settings,
                ij,
                xy,
                boundary_point,
                particle_on)

end

# """
# SeedParticle!(ParticleCollection ::Vector{Any}, State::SharedMatrix, i::Int64,
#                 particle_system::ODESystem, particle_defaults::Union{ParticleDefaultsParam,Nothing}, ODE_settings,
#                 GridNotes, winds, DT:: Float64, Nx:: Int, boundary::Vector{Int}, periodic_boundary::Bool)

# Seed Pickles to ParticleColletion and State
# """
# function SeedParticle!(
#         ParticleCollection::Vector{Any},
#         State::StateTypeL1,
#         ij::Tuple{Int, Int},

#         particle_system::SS,
#         particle_defaults::PP,
#         ODE_settings, #particle_waves_v3.ODESettings type

#         GridNotes, # ad type of grid note
#         winds,     # interp winds
#         DT::Float64,

#         boundary::Vector{T},
#         periodic_boundary::Bool) where {T<:Union{Int,Any,Nothing,Int64},PP<:Union{ParticleDefaultsParam,Nothing},SS<:Union{ODESystem,Any}}

#         # Push Inital condition to collection
#         push!(ParticleCollection,
#                 SeedParticle(State, ij,
#                         particle_system, particle_defaults, ODE_settings, #particle_waves_v3.ODESettings type
#                         GridNotes, winds, DT,
#                         boundary, periodic_boundary))
#         nothing
# end



# end of module
end