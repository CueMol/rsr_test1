
cmake_minimum_required(VERSION 2.8)

find_package(Boost REQUIRED)

##########

find_package(CUDA QUIET)

IF(CUDA_FOUND)
  SET(DEFS ${DEFS} "-DHAVE_CUDA")
  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -gencode arch=compute_30,code=sm_30)
ELSE(CUDA_FOUND)
  message("CUDA NOT FOUND")
ENDIF(CUDA_FOUND)

IF(CUDA_FOUND)
set(CUDA_SRCS
 targ_cuda.cpp 
 com_cuda.cu 
 bond2_cuda.cu 
 targ_cuda_bond2.cpp 
 angl_cuda.cu 
 targ_cuda_angl.cpp 

 map1_cuda.cu 
 map2_cuda.cu 
 targ_cuda_map.cpp 

 plan_cuda.cu 
 targ_cuda_plan.cpp 
 plan2_cuda.cu 
 targ_cuda_nonb.cpp 
 nonb_cuda.cu
)
ENDIF(CUDA_FOUND)

##########

find_path(CUEMOL_SRC_DIR
  NAMES
    qlib/qlib.hpp
  HINTS
    ${CUEMOL_SRC_DIR} ENV CUEMOL_SRC_DIR)

if(UNIX)
find_library(CUEMOL_LIBS
  NAMES
    qlib libqlib
  HINTS
    ${CUEMOL_LIBRARY_DIR} ENV CUEMOL_LIBRARY_DIR)

find_path(CUEMOL_BUILD_DIR
  NAMES
    config.h
  HINTS
    ${CUEMOL_BUILD_DIR} ENV CUEMOL_BUILD_DIR)

set(INCDIRS
  ${INCDIRS}
  ${CUEMOL_SRC_DIR}
  ${CUEMOL_BUILD_DIR})
set(DEFS ${DEFS} -DHAVE_CONFIG_H)
else(UNIX)

find_library(CUEMOL_LIBS_DEBUG
  NAMES
    qlib libqlib
  HINTS
    ${CUEMOL_LIBRARY_DIR} ENV CUEMOL_LIBRARY_DIR
  PATH_SUFFIXES
    xul_Debug64)

find_library(CUEMOL_LIBS_RELEASE
  NAMES
    qlib libqlib
  HINTS
    ${CUEMOL_LIBRARY_DIR} ENV CUEMOL_LIBRARY_DIR
  PATH_SUFFIXES
    xul_Release64)

set(INCDIRS
  ${INCDIRS}
  ${CUEMOL_SRC_DIR})
endif(UNIX)

##########

if(UNIX)
  set(INCDIRS ${INCDIRS} "$ENV{HOME}/app/gsl/include")
  set(LIBS ${LIBS} gsl gslcblas)
  set(GSL_SRCS min_gsl.cpp)
  message("Use GSL")
endif(UNIX)

##########

include_directories(
liblbfgs 
${INCDIRS}
${Boost_INCLUDE_DIRS}
)

link_directories($ENV{HOME}/app/gsl/lib)

add_definitions( ${DEFS} )

cuda_add_executable(program
 main.cpp
 mol.cpp
 minimize.cpp
 targ_cpu.cpp
 targ_cpu_bond.cpp 
 targ_cpu_map.cpp 
 targ_cpu_angl.cpp 
 targ_cpu_dihe.cpp 
 targ_cpu_chir.cpp 
 targ_cpu_plan.cpp 
 targ_cpu_rama.cpp 
 targ_cpu_nonb.cpp 
 RamaPlotData.cpp 
  min_lbgfs.cpp
  liblbfgs/lbfgs.c
 ${GSL_SRCS}
 ${CUDA_SRCS}
)

if(UNIX)
target_link_libraries(program ${CUEMOL_LIBS} ${LIBS} ${Boost_LIBRARIES})
else(UNIX)
target_link_libraries(program debug ${CUEMOL_LIBS_DEBUG} ${LIBS} ${Boost_LIBRARIES})
target_link_libraries(program optimized ${CUEMOL_LIBS_RELEASE} ${LIBS} ${Boost_LIBRARIES})
endif(UNIX)

