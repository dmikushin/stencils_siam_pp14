##===----------------------------------------------------------------------===##
##
##     KernelGen -- A prototype of LLVM-based auto-parallelizing Fortran/C
##        compiler for NVIDIA GPUs, targeting numerical modeling code.
##
## This file is distributed under the University of Illinois Open Source
## License. See LICENSE.TXT for details.
##
##===----------------------------------------------------------------------===##

-include ../../makefile.in

# Verbose setting
VERBOSE ?= 0
SILENT ?= @
ifeq ($(VERBOSE), 1)
SILENT =
endif

# Floating-point type: float or double
real ?= float

# PGI/OpenACC compiler/linker options
GPUARCH ?= 20
GPUCUDA ?= 5.0
NVCC ?= nvcc -I. -I.. -O3 -arch=sm_$(GPUARCH)

# Needed for exact results match
NVCC += --fmad=false

# Default problem domain dimensions
NX ?= 512
NY ?= 256
NS ?= 256
NT ?= 10

all: lbmd3q19_single

lbmd3q19_single: lbmd3q19_single.o timing.o cuda_profiling.o
	$(SILENT)$(NVCC) $^ -o $@ -lrt -Xlinker --wrap=cudaLaunch -Xcompiler -rdynamic

lbmd3q19_single.o: ../lbmd3q19_single.cu ../timing.h cuda_profiling.h
	$(SILENT)$(NVCC) -Dreal=$(real) -c $< -o $@

timing.o: ../timing.c ../timing.h
	$(SILENT)$(NVCC) -c $< -o $@

cuda_profiling.o: cuda_profiling.cu cuda_profiling.h
	$(SILENT)$(NVCC) -c $< -o $@

clean:
	$(SILENT)rm -rf *.o lbmd3q19_single

test: lbmd3q19_single
	$(SILENT)./$< $$(($(NX)/2)) $(NY) $$(($(NS)/2)) $(NT)

check: lbmd3q19_single
	$(SILENT)cuda-memcheck ./$< $$(($(NX)/2)) $(NY) $$(($(NS)/2)) $(NT)

ifeq ($(real), float)
roofline: lbmd3q19_single
	$(SILENT)PROFILING_FNAME=`cat kernel` ./$< $$(($(NX)/2)) $(NY) $$(($(NS)/2)) $(NT) && \
		nvprof --metrics flops_sp,dram_read_throughput,dram_write_throughput ./$< $$(($(NX)/2)) $(NY) $$(($(NS)/2)) $(NT)
endif
ifeq ($(real), double)
roofline: lbmd3q19_single
	$(SILENT)PROFILING_FNAME=`cat kernel` ./$< $$(($(NX)/2)) $(NY) $$(($(NS)/2)) $(NT) && \
		nvprof --metrics flops_sp,dram_read_throughput,dram_write_throughput ./$< $$(($(NX)/2)) $(NY) $$(($(NS)/2)) $(NT)
endif
