using DataFrames, DelimitedFiles, CSV

# Defining helper functions

function unit_dec(a)
    unit=string(Int64(floor((a))))
    dec=string(Int64(floor(10*(a-floor((a))))))
    return unit*","*dec
end

# Starting the merge

parentPath = pwd()*"/plots/test_case_2/model_outputs"
mkdir(parentPath)
mkdir(parentPath*"/data")
mkdir(parentPath*"/plots")
# parentPath = "/home1/datahome/tprotin/Travail/PiCLES/PiCLES/plots/tests/paper/test_case_3/v3_half_cell"
println("------------ Reading the data from : "* parentPath * "  ------------")

nBatch = 10
paths = [parentPath*"_batch"*string(i) for i in 1:nBatch]

path = paths[1]*"/data/simu_infos.csv"
path2 = paths[1]*"/data/sigma.csv"
data, header = readdlm(path, ',', header=true)
data2, header2 = readdlm(path2, ',', header=true)

simu_infos = DataFrame(data, vec(header))
init_sigma_df = DataFrame(data2, vec(header2))

init_sigma = Matrix(init_sigma_df)
Nx = simu_infos.Nx[1]
Ny = simu_infos.Ny[1]
xmin = simu_infos.xmin[1]
xmax = simu_infos.xmax[1]
ymin = simu_infos.ymin[1]
ymax = simu_infos.ymax[1]
lne_source = simu_infos.lne_source[1]
c_x_source = simu_infos.c_x_source[1]
c_y_source = simu_infos.c_y_source[1]
x_source = simu_infos.x_source[1]
y_source = simu_infos.y_source[1]
angular_spread_source = simu_infos.angular_spread_source[1]
Δt = simu_infos.Δt[1]
stop_time = simu_infos.stop_time[1]

nTimes = Int64(ceil(stop_time/Δt))+1

times = (0:(nTimes))*Δt
timesMinutes = times/60

timesMinutesString = unit_dec.(timesMinutes)


ParticleStates = DataFrame([[],[]],["times", "particleList"])
MeshState = DataFrame([[],[]],["times", "mesh_state"])


for i in 1:nTimes
    temp_df1 = DataFrame([[],[],[],[],[],[]], ["id", "logE", "cx", "cy", "x", "y"])
    columnNames = ["Column"*string(Int64(i)) for i in 1:Nx]
    temp_df2 = DataFrame(zeros(70,300), columnNames)
    for j in 1:nBatch
        path1 = paths[j]*"/data/particles_"*timesMinutesString[i]*".csv"
        path2 = paths[j]*"/data/mesh_values_"*timesMinutesString[i]*".csv"
        data1, header1 = readdlm(path1, ',', header=true)
        data2, header2 = readdlm(path2, ',', header=true)

        read_df1 = DataFrame(data1, vec(header1))
        read_df2 = DataFrame(data2, vec(header2))

        append!(temp_df1, read_df1)
        temp_df2 = read_df2 .+ temp_df2
    end
    temp_df1 = DataFrame([[timesMinutes[i]], [temp_df1]], ["times", "particleList"])
    temp_df2 = DataFrame([[timesMinutes[i]], [temp_df2]], ["times", "mesh_state"])
    append!(ParticleStates, temp_df1)
    append!(MeshState, temp_df2)

    time = ParticleStates[end,"times"]*60
    println(time)

    sec=string(Int64(floor((time)/60)))
    dec=string(Int64(floor(10*(time/60-floor((time)/60)))))

    CSV.write(parentPath*"/data/mesh_values_"*sec*","*dec*".csv", temp_df2[end,"mesh_state"])
    CSV.write(parentPath*"/data/particles_"*sec*","*dec*".csv", temp_df1[end,"particleList"])
end
