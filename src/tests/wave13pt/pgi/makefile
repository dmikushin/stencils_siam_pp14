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
PGCC ?= pgcc -I. -acc -O3 -Minfo=accel -ta=nvidia:keep,cc$(GPUARCH),cuda$(GPUCUDA) -Wl,--wrap=__pgi_uacc_cuda_launch
PGF90 ?= pgf90 -acc -O3 -Minfo=accel -ta=nvidia:keep,cc$(GPUARCH),cuda$(GPUCUDA) -Wl,--wrap=__pgi_uacc_cuda_launch $(pgi_szreal)

# NVIDIA compile options
NVCC ?= nvcc -O3

# Default problem domain dimensions
NX ?= 512
NY ?= 256
NS ?= 256
NT ?= 10

all: wave13pt

wave13pt: wave13pt.o timing.o openacc_profiling.o
	$(SILENT)$(PGCC) $^ $(LIBWRAPPER) -o $@ -lrt -L$(shell dirname $(shell which nvcc))/../lib64 -lcudart -lcuda -lstdc++ -lgcc_s

wave13pt.o: ../wave13pt.c ../timing.h
	$(SILENT)$(PGCC) -Dreal=$(real) -c $< -o $@

timing.o: ../timing.c ../timing.h
	$(SILENT)$(PGCC) -c $< -o $@

openacc_profiling.o: openacc_profiling.cu
	$(SILENT)$(NVCC) -c $< -o $@

clean:
	$(SILENT)rm -rf *.o wave13pt *.ptx *.gpu *.bin

test: wave13pt
	$(SILENT)PGI_ACC_TIME=1 ./$< $(NX) $(NY) $(NS) $(NT)

ifeq ($(real), float)
roofline: wave13pt
	$(SILENT)PGI_ACC_TIME=1 ./$< $(NX) $(NY) $(NS) $(NT) && \
		nvprof --metrics flops_sp,dram_read_throughput,dram_write_throughput ./$< $(NX) $(NY) $(NS) $(NT)
endif
ifeq ($(real), double)
roofline: wave13pt
	$(SILENT)PGI_ACC_TIME=1 ./$< $(NX) $(NY) $(NS) $(NT) && \
		nvprof --metrics flops_sp,dram_read_throughput,dram_write_throughput ./$< $(NX) $(NY) $(NS) $(NT)
endif
