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
NVCC ?= nvcc -I. -I.. -O3 -arch=sm_$(GPUARCH) --fmad=false
PPCG ?= ppcg $(pgi_szreal)

# Default problem domain dimensions
NX ?= 512
NY ?= 256
NS ?= 256
NT ?= 10

all: wave13pt

wave13pt: wave13pt_host.o wave13pt_kernel.o timing.o cuda_profiling.o
	$(SILENT)$(NVCC) $^ -o $@ -lrt -Xlinker --wrap=cudaLaunch -Xcompiler -rdynamic

wave13pt_host.cu wave13pt_kernel.cu wave13pt_kernel.hu: ../wave13pt.c ../timing.h cuda_profiling.h
	$(SILENT)$(PPCG) -D_PPCG -Dreal=$(real) $<

wave13pt_host.o: wave13pt_host.cu wave13pt_kernel.hu
	$(SILENT)$(NVCC) -D_PPCG -Dreal=$(real) -c $< -o $@

wave13pt_kernel.o: wave13pt_kernel.cu wave13pt_kernel.hu
	$(SILENT)$(NVCC) -D_PPCG -Dreal=$(real) -c $< -o $@

timing.o: ../timing.c ../timing.h
	$(SILENT)$(NVCC) -c $< -o $@

cuda_profiling.o: cuda_profiling.cu cuda_profiling.h
	$(SILENT)$(NVCC) -c $< -o $@

clean:
	$(SILENT)rm -rf *.o wave13pt wave13pt_host.cu wave13pt_kernel.cu wave13pt_kernel.hu

test: wave13pt
	$(SILENT)./$< $(NX) $(NY) $(NS) $(NT)

check: wave13pt
	$(SILENT)cuda-memcheck ./$< $(NX) $(NY) $(NS) $(NT)

