# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 2.8

# Default target executed when no arguments are given to make.
default_target: all
.PHONY : default_target

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list

# Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The program to use to edit the cache.
CMAKE_EDIT_COMMAND = /usr/bin/ccmake

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/ishitani/proj/cuda/rsr_test1

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/ishitani/proj/cuda/rsr_test1

#=============================================================================
# Targets provided globally by CMake.

# Special rule for the target edit_cache
edit_cache:
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --cyan "Running CMake cache editor..."
	/usr/bin/ccmake -H$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR)
.PHONY : edit_cache

# Special rule for the target edit_cache
edit_cache/fast: edit_cache
.PHONY : edit_cache/fast

# Special rule for the target rebuild_cache
rebuild_cache:
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --cyan "Running CMake to regenerate build system..."
	/usr/bin/cmake -H$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR)
.PHONY : rebuild_cache

# Special rule for the target rebuild_cache
rebuild_cache/fast: rebuild_cache
.PHONY : rebuild_cache/fast

# The main all target
all: cmake_check_build_system
	$(CMAKE_COMMAND) -E cmake_progress_start /home/ishitani/proj/cuda/rsr_test1/CMakeFiles /home/ishitani/proj/cuda/rsr_test1/CMakeFiles/progress.marks
	$(MAKE) -f CMakeFiles/Makefile2 all
	$(CMAKE_COMMAND) -E cmake_progress_start /home/ishitani/proj/cuda/rsr_test1/CMakeFiles 0
.PHONY : all

# The main clean target
clean:
	$(MAKE) -f CMakeFiles/Makefile2 clean
.PHONY : clean

# The main clean target
clean/fast: clean
.PHONY : clean/fast

# Prepare targets for installation.
preinstall: all
	$(MAKE) -f CMakeFiles/Makefile2 preinstall
.PHONY : preinstall

# Prepare targets for installation.
preinstall/fast:
	$(MAKE) -f CMakeFiles/Makefile2 preinstall
.PHONY : preinstall/fast

# clear depends
depend:
	$(CMAKE_COMMAND) -H$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR) --check-build-system CMakeFiles/Makefile.cmake 1
.PHONY : depend

#=============================================================================
# Target rules for targets named program

# Build rule for target.
program: cmake_check_build_system
	$(MAKE) -f CMakeFiles/Makefile2 program
.PHONY : program

# fast build rule for target.
program/fast:
	$(MAKE) -f CMakeFiles/program.dir/build.make CMakeFiles/program.dir/build
.PHONY : program/fast

RamaPlotData.o: RamaPlotData.cpp.o
.PHONY : RamaPlotData.o

# target to build an object file
RamaPlotData.cpp.o:
	$(MAKE) -f CMakeFiles/program.dir/build.make CMakeFiles/program.dir/RamaPlotData.cpp.o
.PHONY : RamaPlotData.cpp.o

RamaPlotData.i: RamaPlotData.cpp.i
.PHONY : RamaPlotData.i

# target to preprocess a source file
RamaPlotData.cpp.i:
	$(MAKE) -f CMakeFiles/program.dir/build.make CMakeFiles/program.dir/RamaPlotData.cpp.i
.PHONY : RamaPlotData.cpp.i

RamaPlotData.s: RamaPlotData.cpp.s
.PHONY : RamaPlotData.s

# target to generate assembly for a file
RamaPlotData.cpp.s:
	$(MAKE) -f CMakeFiles/program.dir/build.make CMakeFiles/program.dir/RamaPlotData.cpp.s
.PHONY : RamaPlotData.cpp.s

liblbfgs/lbfgs.o: liblbfgs/lbfgs.c.o
.PHONY : liblbfgs/lbfgs.o

# target to build an object file
liblbfgs/lbfgs.c.o:
	$(MAKE) -f CMakeFiles/program.dir/build.make CMakeFiles/program.dir/liblbfgs/lbfgs.c.o
.PHONY : liblbfgs/lbfgs.c.o

liblbfgs/lbfgs.i: liblbfgs/lbfgs.c.i
.PHONY : liblbfgs/lbfgs.i

# target to preprocess a source file
liblbfgs/lbfgs.c.i:
	$(MAKE) -f CMakeFiles/program.dir/build.make CMakeFiles/program.dir/liblbfgs/lbfgs.c.i
.PHONY : liblbfgs/lbfgs.c.i

liblbfgs/lbfgs.s: liblbfgs/lbfgs.c.s
.PHONY : liblbfgs/lbfgs.s

# target to generate assembly for a file
liblbfgs/lbfgs.c.s:
	$(MAKE) -f CMakeFiles/program.dir/build.make CMakeFiles/program.dir/liblbfgs/lbfgs.c.s
.PHONY : liblbfgs/lbfgs.c.s

main.o: main.cpp.o
.PHONY : main.o

# target to build an object file
main.cpp.o:
	$(MAKE) -f CMakeFiles/program.dir/build.make CMakeFiles/program.dir/main.cpp.o
.PHONY : main.cpp.o

main.i: main.cpp.i
.PHONY : main.i

# target to preprocess a source file
main.cpp.i:
	$(MAKE) -f CMakeFiles/program.dir/build.make CMakeFiles/program.dir/main.cpp.i
.PHONY : main.cpp.i

main.s: main.cpp.s
.PHONY : main.s

# target to generate assembly for a file
main.cpp.s:
	$(MAKE) -f CMakeFiles/program.dir/build.make CMakeFiles/program.dir/main.cpp.s
.PHONY : main.cpp.s

min_gsl.o: min_gsl.cpp.o
.PHONY : min_gsl.o

# target to build an object file
min_gsl.cpp.o:
	$(MAKE) -f CMakeFiles/program.dir/build.make CMakeFiles/program.dir/min_gsl.cpp.o
.PHONY : min_gsl.cpp.o

min_gsl.i: min_gsl.cpp.i
.PHONY : min_gsl.i

# target to preprocess a source file
min_gsl.cpp.i:
	$(MAKE) -f CMakeFiles/program.dir/build.make CMakeFiles/program.dir/min_gsl.cpp.i
.PHONY : min_gsl.cpp.i

min_gsl.s: min_gsl.cpp.s
.PHONY : min_gsl.s

# target to generate assembly for a file
min_gsl.cpp.s:
	$(MAKE) -f CMakeFiles/program.dir/build.make CMakeFiles/program.dir/min_gsl.cpp.s
.PHONY : min_gsl.cpp.s

min_lbgfs.o: min_lbgfs.cpp.o
.PHONY : min_lbgfs.o

# target to build an object file
min_lbgfs.cpp.o:
	$(MAKE) -f CMakeFiles/program.dir/build.make CMakeFiles/program.dir/min_lbgfs.cpp.o
.PHONY : min_lbgfs.cpp.o

min_lbgfs.i: min_lbgfs.cpp.i
.PHONY : min_lbgfs.i

# target to preprocess a source file
min_lbgfs.cpp.i:
	$(MAKE) -f CMakeFiles/program.dir/build.make CMakeFiles/program.dir/min_lbgfs.cpp.i
.PHONY : min_lbgfs.cpp.i

min_lbgfs.s: min_lbgfs.cpp.s
.PHONY : min_lbgfs.s

# target to generate assembly for a file
min_lbgfs.cpp.s:
	$(MAKE) -f CMakeFiles/program.dir/build.make CMakeFiles/program.dir/min_lbgfs.cpp.s
.PHONY : min_lbgfs.cpp.s

minimize.o: minimize.cpp.o
.PHONY : minimize.o

# target to build an object file
minimize.cpp.o:
	$(MAKE) -f CMakeFiles/program.dir/build.make CMakeFiles/program.dir/minimize.cpp.o
.PHONY : minimize.cpp.o

minimize.i: minimize.cpp.i
.PHONY : minimize.i

# target to preprocess a source file
minimize.cpp.i:
	$(MAKE) -f CMakeFiles/program.dir/build.make CMakeFiles/program.dir/minimize.cpp.i
.PHONY : minimize.cpp.i

minimize.s: minimize.cpp.s
.PHONY : minimize.s

# target to generate assembly for a file
minimize.cpp.s:
	$(MAKE) -f CMakeFiles/program.dir/build.make CMakeFiles/program.dir/minimize.cpp.s
.PHONY : minimize.cpp.s

mol.o: mol.cpp.o
.PHONY : mol.o

# target to build an object file
mol.cpp.o:
	$(MAKE) -f CMakeFiles/program.dir/build.make CMakeFiles/program.dir/mol.cpp.o
.PHONY : mol.cpp.o

mol.i: mol.cpp.i
.PHONY : mol.i

# target to preprocess a source file
mol.cpp.i:
	$(MAKE) -f CMakeFiles/program.dir/build.make CMakeFiles/program.dir/mol.cpp.i
.PHONY : mol.cpp.i

mol.s: mol.cpp.s
.PHONY : mol.s

# target to generate assembly for a file
mol.cpp.s:
	$(MAKE) -f CMakeFiles/program.dir/build.make CMakeFiles/program.dir/mol.cpp.s
.PHONY : mol.cpp.s

targ_cpu.o: targ_cpu.cpp.o
.PHONY : targ_cpu.o

# target to build an object file
targ_cpu.cpp.o:
	$(MAKE) -f CMakeFiles/program.dir/build.make CMakeFiles/program.dir/targ_cpu.cpp.o
.PHONY : targ_cpu.cpp.o

targ_cpu.i: targ_cpu.cpp.i
.PHONY : targ_cpu.i

# target to preprocess a source file
targ_cpu.cpp.i:
	$(MAKE) -f CMakeFiles/program.dir/build.make CMakeFiles/program.dir/targ_cpu.cpp.i
.PHONY : targ_cpu.cpp.i

targ_cpu.s: targ_cpu.cpp.s
.PHONY : targ_cpu.s

# target to generate assembly for a file
targ_cpu.cpp.s:
	$(MAKE) -f CMakeFiles/program.dir/build.make CMakeFiles/program.dir/targ_cpu.cpp.s
.PHONY : targ_cpu.cpp.s

targ_cpu_angl.o: targ_cpu_angl.cpp.o
.PHONY : targ_cpu_angl.o

# target to build an object file
targ_cpu_angl.cpp.o:
	$(MAKE) -f CMakeFiles/program.dir/build.make CMakeFiles/program.dir/targ_cpu_angl.cpp.o
.PHONY : targ_cpu_angl.cpp.o

targ_cpu_angl.i: targ_cpu_angl.cpp.i
.PHONY : targ_cpu_angl.i

# target to preprocess a source file
targ_cpu_angl.cpp.i:
	$(MAKE) -f CMakeFiles/program.dir/build.make CMakeFiles/program.dir/targ_cpu_angl.cpp.i
.PHONY : targ_cpu_angl.cpp.i

targ_cpu_angl.s: targ_cpu_angl.cpp.s
.PHONY : targ_cpu_angl.s

# target to generate assembly for a file
targ_cpu_angl.cpp.s:
	$(MAKE) -f CMakeFiles/program.dir/build.make CMakeFiles/program.dir/targ_cpu_angl.cpp.s
.PHONY : targ_cpu_angl.cpp.s

targ_cpu_bond.o: targ_cpu_bond.cpp.o
.PHONY : targ_cpu_bond.o

# target to build an object file
targ_cpu_bond.cpp.o:
	$(MAKE) -f CMakeFiles/program.dir/build.make CMakeFiles/program.dir/targ_cpu_bond.cpp.o
.PHONY : targ_cpu_bond.cpp.o

targ_cpu_bond.i: targ_cpu_bond.cpp.i
.PHONY : targ_cpu_bond.i

# target to preprocess a source file
targ_cpu_bond.cpp.i:
	$(MAKE) -f CMakeFiles/program.dir/build.make CMakeFiles/program.dir/targ_cpu_bond.cpp.i
.PHONY : targ_cpu_bond.cpp.i

targ_cpu_bond.s: targ_cpu_bond.cpp.s
.PHONY : targ_cpu_bond.s

# target to generate assembly for a file
targ_cpu_bond.cpp.s:
	$(MAKE) -f CMakeFiles/program.dir/build.make CMakeFiles/program.dir/targ_cpu_bond.cpp.s
.PHONY : targ_cpu_bond.cpp.s

targ_cpu_chir.o: targ_cpu_chir.cpp.o
.PHONY : targ_cpu_chir.o

# target to build an object file
targ_cpu_chir.cpp.o:
	$(MAKE) -f CMakeFiles/program.dir/build.make CMakeFiles/program.dir/targ_cpu_chir.cpp.o
.PHONY : targ_cpu_chir.cpp.o

targ_cpu_chir.i: targ_cpu_chir.cpp.i
.PHONY : targ_cpu_chir.i

# target to preprocess a source file
targ_cpu_chir.cpp.i:
	$(MAKE) -f CMakeFiles/program.dir/build.make CMakeFiles/program.dir/targ_cpu_chir.cpp.i
.PHONY : targ_cpu_chir.cpp.i

targ_cpu_chir.s: targ_cpu_chir.cpp.s
.PHONY : targ_cpu_chir.s

# target to generate assembly for a file
targ_cpu_chir.cpp.s:
	$(MAKE) -f CMakeFiles/program.dir/build.make CMakeFiles/program.dir/targ_cpu_chir.cpp.s
.PHONY : targ_cpu_chir.cpp.s

targ_cpu_dihe.o: targ_cpu_dihe.cpp.o
.PHONY : targ_cpu_dihe.o

# target to build an object file
targ_cpu_dihe.cpp.o:
	$(MAKE) -f CMakeFiles/program.dir/build.make CMakeFiles/program.dir/targ_cpu_dihe.cpp.o
.PHONY : targ_cpu_dihe.cpp.o

targ_cpu_dihe.i: targ_cpu_dihe.cpp.i
.PHONY : targ_cpu_dihe.i

# target to preprocess a source file
targ_cpu_dihe.cpp.i:
	$(MAKE) -f CMakeFiles/program.dir/build.make CMakeFiles/program.dir/targ_cpu_dihe.cpp.i
.PHONY : targ_cpu_dihe.cpp.i

targ_cpu_dihe.s: targ_cpu_dihe.cpp.s
.PHONY : targ_cpu_dihe.s

# target to generate assembly for a file
targ_cpu_dihe.cpp.s:
	$(MAKE) -f CMakeFiles/program.dir/build.make CMakeFiles/program.dir/targ_cpu_dihe.cpp.s
.PHONY : targ_cpu_dihe.cpp.s

targ_cpu_map.o: targ_cpu_map.cpp.o
.PHONY : targ_cpu_map.o

# target to build an object file
targ_cpu_map.cpp.o:
	$(MAKE) -f CMakeFiles/program.dir/build.make CMakeFiles/program.dir/targ_cpu_map.cpp.o
.PHONY : targ_cpu_map.cpp.o

targ_cpu_map.i: targ_cpu_map.cpp.i
.PHONY : targ_cpu_map.i

# target to preprocess a source file
targ_cpu_map.cpp.i:
	$(MAKE) -f CMakeFiles/program.dir/build.make CMakeFiles/program.dir/targ_cpu_map.cpp.i
.PHONY : targ_cpu_map.cpp.i

targ_cpu_map.s: targ_cpu_map.cpp.s
.PHONY : targ_cpu_map.s

# target to generate assembly for a file
targ_cpu_map.cpp.s:
	$(MAKE) -f CMakeFiles/program.dir/build.make CMakeFiles/program.dir/targ_cpu_map.cpp.s
.PHONY : targ_cpu_map.cpp.s

targ_cpu_nonb.o: targ_cpu_nonb.cpp.o
.PHONY : targ_cpu_nonb.o

# target to build an object file
targ_cpu_nonb.cpp.o:
	$(MAKE) -f CMakeFiles/program.dir/build.make CMakeFiles/program.dir/targ_cpu_nonb.cpp.o
.PHONY : targ_cpu_nonb.cpp.o

targ_cpu_nonb.i: targ_cpu_nonb.cpp.i
.PHONY : targ_cpu_nonb.i

# target to preprocess a source file
targ_cpu_nonb.cpp.i:
	$(MAKE) -f CMakeFiles/program.dir/build.make CMakeFiles/program.dir/targ_cpu_nonb.cpp.i
.PHONY : targ_cpu_nonb.cpp.i

targ_cpu_nonb.s: targ_cpu_nonb.cpp.s
.PHONY : targ_cpu_nonb.s

# target to generate assembly for a file
targ_cpu_nonb.cpp.s:
	$(MAKE) -f CMakeFiles/program.dir/build.make CMakeFiles/program.dir/targ_cpu_nonb.cpp.s
.PHONY : targ_cpu_nonb.cpp.s

targ_cpu_plan.o: targ_cpu_plan.cpp.o
.PHONY : targ_cpu_plan.o

# target to build an object file
targ_cpu_plan.cpp.o:
	$(MAKE) -f CMakeFiles/program.dir/build.make CMakeFiles/program.dir/targ_cpu_plan.cpp.o
.PHONY : targ_cpu_plan.cpp.o

targ_cpu_plan.i: targ_cpu_plan.cpp.i
.PHONY : targ_cpu_plan.i

# target to preprocess a source file
targ_cpu_plan.cpp.i:
	$(MAKE) -f CMakeFiles/program.dir/build.make CMakeFiles/program.dir/targ_cpu_plan.cpp.i
.PHONY : targ_cpu_plan.cpp.i

targ_cpu_plan.s: targ_cpu_plan.cpp.s
.PHONY : targ_cpu_plan.s

# target to generate assembly for a file
targ_cpu_plan.cpp.s:
	$(MAKE) -f CMakeFiles/program.dir/build.make CMakeFiles/program.dir/targ_cpu_plan.cpp.s
.PHONY : targ_cpu_plan.cpp.s

targ_cpu_rama.o: targ_cpu_rama.cpp.o
.PHONY : targ_cpu_rama.o

# target to build an object file
targ_cpu_rama.cpp.o:
	$(MAKE) -f CMakeFiles/program.dir/build.make CMakeFiles/program.dir/targ_cpu_rama.cpp.o
.PHONY : targ_cpu_rama.cpp.o

targ_cpu_rama.i: targ_cpu_rama.cpp.i
.PHONY : targ_cpu_rama.i

# target to preprocess a source file
targ_cpu_rama.cpp.i:
	$(MAKE) -f CMakeFiles/program.dir/build.make CMakeFiles/program.dir/targ_cpu_rama.cpp.i
.PHONY : targ_cpu_rama.cpp.i

targ_cpu_rama.s: targ_cpu_rama.cpp.s
.PHONY : targ_cpu_rama.s

# target to generate assembly for a file
targ_cpu_rama.cpp.s:
	$(MAKE) -f CMakeFiles/program.dir/build.make CMakeFiles/program.dir/targ_cpu_rama.cpp.s
.PHONY : targ_cpu_rama.cpp.s

targ_cuda.o: targ_cuda.cpp.o
.PHONY : targ_cuda.o

# target to build an object file
targ_cuda.cpp.o:
	$(MAKE) -f CMakeFiles/program.dir/build.make CMakeFiles/program.dir/targ_cuda.cpp.o
.PHONY : targ_cuda.cpp.o

targ_cuda.i: targ_cuda.cpp.i
.PHONY : targ_cuda.i

# target to preprocess a source file
targ_cuda.cpp.i:
	$(MAKE) -f CMakeFiles/program.dir/build.make CMakeFiles/program.dir/targ_cuda.cpp.i
.PHONY : targ_cuda.cpp.i

targ_cuda.s: targ_cuda.cpp.s
.PHONY : targ_cuda.s

# target to generate assembly for a file
targ_cuda.cpp.s:
	$(MAKE) -f CMakeFiles/program.dir/build.make CMakeFiles/program.dir/targ_cuda.cpp.s
.PHONY : targ_cuda.cpp.s

targ_cuda_angl.o: targ_cuda_angl.cpp.o
.PHONY : targ_cuda_angl.o

# target to build an object file
targ_cuda_angl.cpp.o:
	$(MAKE) -f CMakeFiles/program.dir/build.make CMakeFiles/program.dir/targ_cuda_angl.cpp.o
.PHONY : targ_cuda_angl.cpp.o

targ_cuda_angl.i: targ_cuda_angl.cpp.i
.PHONY : targ_cuda_angl.i

# target to preprocess a source file
targ_cuda_angl.cpp.i:
	$(MAKE) -f CMakeFiles/program.dir/build.make CMakeFiles/program.dir/targ_cuda_angl.cpp.i
.PHONY : targ_cuda_angl.cpp.i

targ_cuda_angl.s: targ_cuda_angl.cpp.s
.PHONY : targ_cuda_angl.s

# target to generate assembly for a file
targ_cuda_angl.cpp.s:
	$(MAKE) -f CMakeFiles/program.dir/build.make CMakeFiles/program.dir/targ_cuda_angl.cpp.s
.PHONY : targ_cuda_angl.cpp.s

targ_cuda_bond2.o: targ_cuda_bond2.cpp.o
.PHONY : targ_cuda_bond2.o

# target to build an object file
targ_cuda_bond2.cpp.o:
	$(MAKE) -f CMakeFiles/program.dir/build.make CMakeFiles/program.dir/targ_cuda_bond2.cpp.o
.PHONY : targ_cuda_bond2.cpp.o

targ_cuda_bond2.i: targ_cuda_bond2.cpp.i
.PHONY : targ_cuda_bond2.i

# target to preprocess a source file
targ_cuda_bond2.cpp.i:
	$(MAKE) -f CMakeFiles/program.dir/build.make CMakeFiles/program.dir/targ_cuda_bond2.cpp.i
.PHONY : targ_cuda_bond2.cpp.i

targ_cuda_bond2.s: targ_cuda_bond2.cpp.s
.PHONY : targ_cuda_bond2.s

# target to generate assembly for a file
targ_cuda_bond2.cpp.s:
	$(MAKE) -f CMakeFiles/program.dir/build.make CMakeFiles/program.dir/targ_cuda_bond2.cpp.s
.PHONY : targ_cuda_bond2.cpp.s

targ_cuda_map.o: targ_cuda_map.cpp.o
.PHONY : targ_cuda_map.o

# target to build an object file
targ_cuda_map.cpp.o:
	$(MAKE) -f CMakeFiles/program.dir/build.make CMakeFiles/program.dir/targ_cuda_map.cpp.o
.PHONY : targ_cuda_map.cpp.o

targ_cuda_map.i: targ_cuda_map.cpp.i
.PHONY : targ_cuda_map.i

# target to preprocess a source file
targ_cuda_map.cpp.i:
	$(MAKE) -f CMakeFiles/program.dir/build.make CMakeFiles/program.dir/targ_cuda_map.cpp.i
.PHONY : targ_cuda_map.cpp.i

targ_cuda_map.s: targ_cuda_map.cpp.s
.PHONY : targ_cuda_map.s

# target to generate assembly for a file
targ_cuda_map.cpp.s:
	$(MAKE) -f CMakeFiles/program.dir/build.make CMakeFiles/program.dir/targ_cuda_map.cpp.s
.PHONY : targ_cuda_map.cpp.s

targ_cuda_nonb.o: targ_cuda_nonb.cpp.o
.PHONY : targ_cuda_nonb.o

# target to build an object file
targ_cuda_nonb.cpp.o:
	$(MAKE) -f CMakeFiles/program.dir/build.make CMakeFiles/program.dir/targ_cuda_nonb.cpp.o
.PHONY : targ_cuda_nonb.cpp.o

targ_cuda_nonb.i: targ_cuda_nonb.cpp.i
.PHONY : targ_cuda_nonb.i

# target to preprocess a source file
targ_cuda_nonb.cpp.i:
	$(MAKE) -f CMakeFiles/program.dir/build.make CMakeFiles/program.dir/targ_cuda_nonb.cpp.i
.PHONY : targ_cuda_nonb.cpp.i

targ_cuda_nonb.s: targ_cuda_nonb.cpp.s
.PHONY : targ_cuda_nonb.s

# target to generate assembly for a file
targ_cuda_nonb.cpp.s:
	$(MAKE) -f CMakeFiles/program.dir/build.make CMakeFiles/program.dir/targ_cuda_nonb.cpp.s
.PHONY : targ_cuda_nonb.cpp.s

targ_cuda_plan.o: targ_cuda_plan.cpp.o
.PHONY : targ_cuda_plan.o

# target to build an object file
targ_cuda_plan.cpp.o:
	$(MAKE) -f CMakeFiles/program.dir/build.make CMakeFiles/program.dir/targ_cuda_plan.cpp.o
.PHONY : targ_cuda_plan.cpp.o

targ_cuda_plan.i: targ_cuda_plan.cpp.i
.PHONY : targ_cuda_plan.i

# target to preprocess a source file
targ_cuda_plan.cpp.i:
	$(MAKE) -f CMakeFiles/program.dir/build.make CMakeFiles/program.dir/targ_cuda_plan.cpp.i
.PHONY : targ_cuda_plan.cpp.i

targ_cuda_plan.s: targ_cuda_plan.cpp.s
.PHONY : targ_cuda_plan.s

# target to generate assembly for a file
targ_cuda_plan.cpp.s:
	$(MAKE) -f CMakeFiles/program.dir/build.make CMakeFiles/program.dir/targ_cuda_plan.cpp.s
.PHONY : targ_cuda_plan.cpp.s

# Help Target
help:
	@echo "The following are some of the valid targets for this Makefile:"
	@echo "... all (the default if no target is provided)"
	@echo "... clean"
	@echo "... depend"
	@echo "... edit_cache"
	@echo "... program"
	@echo "... rebuild_cache"
	@echo "... RamaPlotData.o"
	@echo "... RamaPlotData.i"
	@echo "... RamaPlotData.s"
	@echo "... liblbfgs/lbfgs.o"
	@echo "... liblbfgs/lbfgs.i"
	@echo "... liblbfgs/lbfgs.s"
	@echo "... main.o"
	@echo "... main.i"
	@echo "... main.s"
	@echo "... min_gsl.o"
	@echo "... min_gsl.i"
	@echo "... min_gsl.s"
	@echo "... min_lbgfs.o"
	@echo "... min_lbgfs.i"
	@echo "... min_lbgfs.s"
	@echo "... minimize.o"
	@echo "... minimize.i"
	@echo "... minimize.s"
	@echo "... mol.o"
	@echo "... mol.i"
	@echo "... mol.s"
	@echo "... targ_cpu.o"
	@echo "... targ_cpu.i"
	@echo "... targ_cpu.s"
	@echo "... targ_cpu_angl.o"
	@echo "... targ_cpu_angl.i"
	@echo "... targ_cpu_angl.s"
	@echo "... targ_cpu_bond.o"
	@echo "... targ_cpu_bond.i"
	@echo "... targ_cpu_bond.s"
	@echo "... targ_cpu_chir.o"
	@echo "... targ_cpu_chir.i"
	@echo "... targ_cpu_chir.s"
	@echo "... targ_cpu_dihe.o"
	@echo "... targ_cpu_dihe.i"
	@echo "... targ_cpu_dihe.s"
	@echo "... targ_cpu_map.o"
	@echo "... targ_cpu_map.i"
	@echo "... targ_cpu_map.s"
	@echo "... targ_cpu_nonb.o"
	@echo "... targ_cpu_nonb.i"
	@echo "... targ_cpu_nonb.s"
	@echo "... targ_cpu_plan.o"
	@echo "... targ_cpu_plan.i"
	@echo "... targ_cpu_plan.s"
	@echo "... targ_cpu_rama.o"
	@echo "... targ_cpu_rama.i"
	@echo "... targ_cpu_rama.s"
	@echo "... targ_cuda.o"
	@echo "... targ_cuda.i"
	@echo "... targ_cuda.s"
	@echo "... targ_cuda_angl.o"
	@echo "... targ_cuda_angl.i"
	@echo "... targ_cuda_angl.s"
	@echo "... targ_cuda_bond2.o"
	@echo "... targ_cuda_bond2.i"
	@echo "... targ_cuda_bond2.s"
	@echo "... targ_cuda_map.o"
	@echo "... targ_cuda_map.i"
	@echo "... targ_cuda_map.s"
	@echo "... targ_cuda_nonb.o"
	@echo "... targ_cuda_nonb.i"
	@echo "... targ_cuda_nonb.s"
	@echo "... targ_cuda_plan.o"
	@echo "... targ_cuda_plan.i"
	@echo "... targ_cuda_plan.s"
.PHONY : help



#=============================================================================
# Special targets to cleanup operation of make.

# Special rule to run CMake to check the build system integrity.
# No rule that depends on this can have commands that come from listfiles
# because they might be regenerated.
cmake_check_build_system:
	$(CMAKE_COMMAND) -H$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR) --check-build-system CMakeFiles/Makefile.cmake 0
.PHONY : cmake_check_build_system

