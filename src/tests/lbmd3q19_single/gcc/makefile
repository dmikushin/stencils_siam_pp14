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

# GCC compiler/linker options
GCC ?= gcc -O3 -D_GNU_SOURCE -std=c99
GXX ?= g++ -O3 -D_GNU_SOURCE -fopenmp -ftree-vectorize -mtune=native

# Default problem domain dimensions
NX ?= 512
NY ?= 256
NS ?= 256
NT ?= 10

all: lbmd3q19_single

lbmd3q19_single: lbmd3q19_single.o timing.o
	$(SILENT)$(GXX) $^ -o $@ -lrt -lm

lbmd3q19_single.o: ../lbmd3q19_single.cpp ../genlbm.h ../timing.h
	$(SILENT)$(GXX) -Dreal=$(real) -c $< -o $@

timing.o: ../timing.c ../timing.h
	$(SILENT)$(GCC) -c $< -o $@

clean:
	$(SILENT)rm -rf *.o lbmd3q19_single

test: lbmd3q19_single
	$(SILENT)./$< $$(($(NX)/2)) $(NY) $$(($(NS)/2)) $(NT)
