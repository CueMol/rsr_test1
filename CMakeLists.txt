
cmake_minimum_required(VERSION 2.8)

##########

find_package(CUDA QUIET)

IF(CUDA_FOUND)
  SET(DEFS ${DEFS} "-DHAVE_CUDA")
  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -gencode arch=compute_30,code=sm_30)
ELSE(CUDA_FOUND)
  message("CUDA NOT FOUND")
ENDIF(CUDA_FOUND)

##########

find_library(CUEMOL_LIBS
  NAMES
    qlib libqlib
  HINTS
    ${CUEMOL_LIBRARY_DIR} ENV CUEMOL_LIBRARY_DIR)

find_path(CUEMOL_SRC_DIR
  NAMES
    qlib/qlib.hpp
  HINTS
    ${CUEMOL_SRC_DIR} ENV CUEMOL_SRC_DIR)

find_path(CUEMOL_BUILD_DIR
  NAMES
    config.h
  HINTS
    ${CUEMOL_BUILD_DIR} ENV CUEMOL_BUILD_DIR)

##########

if(UNIX)
  set(DEFS ${DEFS} "-DHAVE_GSL")
  set(INCDIRS ${INCDIRS} "$ENV{HOME}/app/gsl/include")
  set(LIBS ${LIBS} gsl gslcblas)
  message("Use GSL")
endif(UNIX)

include_directories(
liblbfgs 
${CUEMOL_SRC_DIR}
${CUEMOL_BUILD_DIR}
${INCDIRS}
)


link_directories($ENV{HOME}/app/gsl/lib)

add_definitions( ${DEFS} -DHAVE_CONFIG_H )

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

target_link_libraries(program ${CUEMOL_LIBS} ${LIBS})

