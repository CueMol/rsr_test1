#
#
#

NVCC=nvcc

NVCCFLAGS=-gencode arch=compute_30,code=sm_30
#NVCCFLAGS=--ftz=true --prec-div=false --prec-sqrt=false --fmad=true
#NVCCFLAGS=--ftz=false --prec-div=true --prec-sqrt=true --fmad=false

CUDA_LDFLAGS= \
  -Wl,-rpath /usr/local/cuda/lib64 \
  -L/usr/local/cuda/lib64 \
  -lcuda -lcudart

OBJS=main.o \
 mol.o \
 minimize.o \
 min_gsl.o \
 min_lbgfs.o \
 liblbfgs/lbfgs.o \
 targ_cpu.o \
 targ_cpu_bond.o \
 targ_cpu_map.o \
 targ_cpu_angl.o \
 targ_cpu_dihe.o \
 targ_cpu_chir.o \
 targ_cpu_plan.o \
 targ_cpu_rama.o \
 targ_cpu_nonb.o \
 RamaPlotData.o \
 \
 targ_cuda.o \
 com_cuda.o \
 bond_cuda.o \
 targ_cuda_bond.o \
 angl_cuda.o \
 targ_cuda_angl.o \
 map1_cuda.o \
 targ_cuda_map.o \
 map2_cuda.o \
 plan_cuda.o \
 targ_cuda_plan.o \
$(NULL)

# targ_cuda.o

CFLAGS=-g \
  -DHAVE_CUDA \
  -DHAVE_GSL \
  -Iliblbfgs \
  -I$(HOME)/proj/cuemol2_uxcli/src \
  -I$(HOME)/proj/cuemol2_uxcli/uxbuild \
  -I$(HOME)/app/gsl/include


CXXFLAGS= $(CFLAGS) \
  -std=c++0x -DHAVE_CONFIG_H \

LDFLAGS= \
  -Wl,-rpath $(HOME)/app/cuemol2/lib \
  -L$(HOME)/app/cuemol2/lib \
  -Wl,-rpath $(HOME)/app/gsl/lib \
  -L$(HOME)/app/gsl/lib 

LDADD=-lqlib \
-lgsl -lgslcblas

%_cuda.o : %_cuda.cu
	$(NVCC) $(NVCCFLAGS) -c $<

all: program

program: $(OBJS)
	g++ -o program $(OBJS) $(LDFLAGS) $(LDADD) $(CUDA_LDFLAGS)

#program: cudacode.o main.o
#    g++ -o program -L/usr/local/cuda/lib64 -lcuda -lcudart main.o cudacode.o 

cuda.o: cuda.cu cuda.hpp

#cuda_map.o: cuda_map.cu cudacode.h cuda_map_kern.hpp cuda_map_kern2.hpp
#	nvcc $(NVCCFLAGS) -c cuda_map.cu 

##

clean:
	rm -rf *.o program

main.o: main.cpp map.hpp mol.hpp grad_bond.hpp grad_map.hpp

minimize.o: minimize.cpp map.hpp mol.hpp

targ_cuda.o: targ_cuda.cpp map.hpp mol.hpp

targ_cpu.o: targ_cpu.cpp map.hpp mol.hpp minimize.hpp

targ_cpu_bond.o: targ_cpu_bond.cpp mol.hpp minimize.hpp

targ_cpu_angl.o: targ_cpu_angl.cpp mol.hpp minimize.hpp

targ_cpu_dihe.o: targ_cpu_dihe.cpp mol.hpp minimize.hpp

targ_cpu_chir.o: targ_cpu_chir.cpp mol.hpp minimize.hpp

targ_cpu_plan.o: targ_cpu_plan.cpp mol.hpp minimize.hpp mat33_diag.hpp

targ_cpu_rama.o: targ_cpu_rama.cpp mol.hpp minimize.hpp

min_gsl.o: min_gsl.cpp map.hpp mol.hpp

min_lbfgs.o: min_lbfgs.cpp minimize.hpp map.hpp mol.hpp


