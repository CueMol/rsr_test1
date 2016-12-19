
cmake_minimum_required(VERSION 2.8)
find_package(CUDA REQUIRED)

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -gencode arch=compute_30,code=sm_30)

include_directories(
liblbfgs 
$ENV{HOME}/proj/cuemol2_uxcli/src 
$ENV{HOME}/proj/cuemol2_uxcli/uxbuild 
$ENV{HOME}/app/gsl/include
)

add_definitions( -DHAVE_CUDA -DHAVE_GSL -DHAVE_CONFIG_H )

link_directories($ENV{HOME}/app/cuemol2/lib $ENV{HOME}/app/gsl/lib)

cuda_add_executable(program main.cpp  mol.cpp  minimize.cpp  min_gsl.cpp  min_lbgfs.cpp  liblbfgs/lbfgs.c  targ_cpu.cpp  targ_cpu_bond.cpp 
 targ_cpu_map.cpp 
 targ_cpu_angl.cpp 
 targ_cpu_dihe.cpp 
 targ_cpu_chir.cpp 
 targ_cpu_plan.cpp 
 targ_cpu_rama.cpp 
 targ_cpu_nonb.cpp 
 RamaPlotData.cpp 
  targ_cuda.cpp 
 com_cuda.cu 
 bond2_cuda.cu 
 targ_cuda_bond2.cpp 
 angl_cuda.cu 
 targ_cuda_angl.cpp 
 map1_cuda.cu 
 targ_cuda_map.cpp 
 map2_cuda.cu 
 plan_cuda.cu 
 targ_cuda_plan.cpp 
 plan2_cuda.cu 
 targ_cuda_nonb.cpp 
 nonb_cuda.cu )

target_link_libraries(program qlib gsl gslcblas)

