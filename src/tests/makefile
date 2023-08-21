##===----------------------------------------------------------------------===##
##
##     KernelGen -- A prototype of LLVM-based auto-parallelizing Fortran/C
##        compiler for NVIDIA GPUs, targeting numerical modeling code.
##
## This file is distributed under the University of Illinois Open Source
## License. See LICENSE.TXT for details.
##
##===----------------------------------------------------------------------===##

-include makefile.in

TARGETS = \
	  wave13pt

TARGETS_GCC = $(addsuffix .gcc, $(TARGETS))
TARGETS_GCC_CLEAN = $(addsuffix .gcc.clean, $(TARGETS))

TARGETS_KERNELGEN = $(addsuffix .kernelgen, $(TARGETS))
TARGETS_KERNELGEN_CLEAN = $(addsuffix .kernelgen.clean, $(TARGETS))

TARGETS_PGI = $(addsuffix .pgi, $(TARGETS))
TARGETS_PGI_CLEAN = $(addsuffix .pgi.clean, $(TARGETS))

TARGETS_PPCG = $(addsuffix .ppcg, $(TARGETS))
TARGETS_PPCG_CLEAN = $(addsuffix .ppcg.clean, $(TARGETS))

TARGETS_CUDA = $(addsuffix .cuda, $(TARGETS))
TARGETS_CUDA_CHECK = $(addsuffix .cuda.check, $(TARGETS))

TARGETS_CUDA_SHFL = $(addsuffix .cuda_shfl, $(TARGETS))
TARGETS_CUDA_SHMEM1DREG1D = $(addsuffix .cuda_shmem1dreg1d, $(TARGETS))
TARGETS_CUDA_SHMEM1DREG1DVEC2 = $(addsuffix .cuda_shmem1dreg1dvec2, $(TARGETS))
TARGETS_CUDA_SHMEM1DVEC2 = $(addsuffix .cuda_shmem1dvec2, $(TARGETS))
TARGETS_CUDA_SHMEM2D = $(addsuffix .cuda_shmem2d, $(TARGETS))
TARGETS_CUDA_SHMEM2DREG1D = $(addsuffix .cuda_shmem2dreg1d, $(TARGETS))
TARGETS_CUDA_SHMEM2DREG1DVEC2 = $(addsuffix .cuda_shmem2dreg1dvec2, $(TARGETS))
TARGETS_CUDA_SHMEM2DVEC2 = $(addsuffix .cuda_shmem2dvec2, $(TARGETS))
TARGETS_CUDA_VEC2 = $(addsuffix .cuda_vec2, $(TARGETS))
TARGETS_CUDA_VEC2SHFL = $(addsuffix .cuda_vec2shfl, $(TARGETS))

TARGETS_CUDA_SHFL_CHECK = $(addsuffix .cuda_shfl.check, $(TARGETS))
TARGETS_CUDA_SHMEM1DREG1D_CHECK = $(addsuffix .cuda_shmem1dreg1d.check, $(TARGETS))
TARGETS_CUDA_SHMEM1DREG1DVEC2_CHECK = $(addsuffix .cuda_shmem1dreg1dvec2.check, $(TARGETS))
TARGETS_CUDA_SHMEM1DVEC2_CHECK = $(addsuffix .cuda_shmem1dvec2.check, $(TARGETS))
TARGETS_CUDA_SHMEM2D_CHECK = $(addsuffix .cuda_shmem2d.check, $(TARGETS))
TARGETS_CUDA_SHMEM2DREG1D_CHECK = $(addsuffix .cuda_shmem2dreg1d.check, $(TARGETS))
TARGETS_CUDA_SHMEM2DREG1DVEC2_CHECK = $(addsuffix .cuda_shmem2dreg1dvec2.check, $(TARGETS))
TARGETS_CUDA_SHMEM2DVEC2_CHECK = $(addsuffix .cuda_shmem2dvec2.check, $(TARGETS))
TARGETS_CUDA_VEC2_CHECK = $(addsuffix .cuda_vec2.check, $(TARGETS))
TARGETS_CUDA_VEC2SHFL_CHECK = $(addsuffix .cuda_vec2shfl.check, $(TARGETS))

TARGETS_CUDA_SHFL_CLEAN = $(addsuffix .cuda_shfl.clean, $(TARGETS))
TARGETS_CUDA_SHMEM1DREG1D_CLEAN = $(addsuffix .cuda_shmem1dreg1d.clean, $(TARGETS))
TARGETS_CUDA_SHMEM1DREG1DVEC2_CLEAN = $(addsuffix .cuda_shmem1dreg1dvec2.clean, $(TARGETS))
TARGETS_CUDA_SHMEM1DVEC2_CLEAN = $(addsuffix .cuda_shmem1dvec2.clean, $(TARGETS))
TARGETS_CUDA_SHMEM2D_CLEAN = $(addsuffix .cuda_shmem2d.clean, $(TARGETS))
TARGETS_CUDA_SHMEM2DREG1D_CLEAN = $(addsuffix .cuda_shmem2dreg1d.clean, $(TARGETS))
TARGETS_CUDA_SHMEM2DREG1DVEC2_CLEAN = $(addsuffix .cuda_shmem2dreg1dvec2.clean, $(TARGETS))
TARGETS_CUDA_SHMEM2DVEC2_CLEAN = $(addsuffix .cuda_shmem2dvec2.clean, $(TARGETS))
TARGETS_CUDA_VEC2_CLEAN = $(addsuffix .cuda_vec2.clean, $(TARGETS))
TARGETS_CUDA_VEC2SHFL_CLEAN = $(addsuffix .cuda_vec2shfl.clean, $(TARGETS))

TARGETS_CUDA_ALL = $(TARGETS_CUDA) $(TARGETS_CUDA_SHFL) $(TARGETS_CUDA_SHMEM1D) \
	$(TARGETS_CUDA_SHMEM1DREG1D) $(TARGETS_CUDA_SHMEM1DREG1DVEC2) \
	$(TARGETS_CUDA_SHMEM1DVEC2) $(TARGETS_CUDA_SHMEM2D) $(TARGETS_CUDA_SHMEM2DREG1D) \
	$(TARGETS_CUDA_SHMEM2DREG1DVEC2) $(TARGETS_CUDA_SHMEM2DVEC2) $(TARGETS_CUDA_VEC2) \
	$(TARGETS_CUDA_VEC2SHFL)
TARGETS_CUDA_ALL_CHECK = $(TARGETS_CUDA_CHECK) $(TARGETS_CUDA_SHFL_CHECK) $(TARGETS_CUDA_SHMEM1D_CHECK) \
	$(TARGETS_CUDA_SHMEM1DREG1D_CHECK) $(TARGETS_CUDA_SHMEM1DREG1DVEC2_CHECK) \
	$(TARGETS_CUDA_SHMEM1DVEC2_CHECK) $(TARGETS_CUDA_SHMEM2D_CHECK) $(TARGETS_CUDA_SHMEM2DREG1D_CHECK) \
	$(TARGETS_CUDA_SHMEM2DREG1DVEC2_CHECK) $(TARGETS_CUDA_SHMEM2DVEC2_CHECK) $(TARGETS_CUDA_VEC2_CHECK) \
	$(TARGETS_CUDA_VEC2SHFL_CHECK)
TARGETS_CUDA_ALL_CLEAN = $(TARGETS_CUDA_CLEAN) $(TARGETS_CUDA_SHFL_CLEAN) $(TARGETS_CUDA_SHMEM1D_CLEAN) \
	$(TARGETS_CUDA_SHMEM1DREG1D_CLEAN) $(TARGETS_CUDA_SHMEM1DREG1DVEC2_CLEAN) \
	$(TARGETS_CUDA_SHMEM1DVEC2_CLEAN) $(TARGETS_CUDA_SHMEM2D_CLEAN) $(TARGETS_CUDA_SHMEM2DREG1D_CLEAN) \
	$(TARGETS_CUDA_SHMEM2DREG1DVEC2_CLEAN) $(TARGETS_CUDA_SHMEM2DVEC2_CLEAN) $(TARGETS_CUDA_VEC2_CLEAN) \
	$(TARGETS_CUDA_VEC2SHFL_CLEAN)

.PHONY: all

all: gcc pgi cuda_all

gcc: $(TARGETS_GCC)

kernelgen: $(TARGETS_KERNELGEN)

pgi: $(TARGETS_PGI)

ppcg: $(TARGETS_PPCG)

cuda_all: $(TARGETS_CUDA_ALL)

cuda_check_all: $(TARGETS_CUDA_ALL_CHECK)

cuda: $(TARGETS_CUDA)

clean: $(TARGETS_GCC_CLEAN) $(TARGETS_KERNELGEN_CLEAN) $(TARGETS_PGI_CLEAN) $(TARGETS_PPCG_CLEAN) $(TARGETS_CUDA_ALL_CLEAN)

%.gcc:
	$(SILENT)cd $(subst .gcc,,$@)/gcc && $(MAKE)
	
%.gcc.clean:
	$(SILENT)cd $(subst .gcc.clean,,$@)/gcc && $(MAKE) clean

%.kernelgen:
	$(SILENT)cd $(subst .kernelgen,,$@)/kernelgen && $(MAKE)

%.kernelgen.clean:
	$(SILENT)cd $(subst .kernelgen.clean,,$@)/kernelgen && $(MAKE) clean

%.pgi:
	$(SILENT)cd $(subst .pgi,,$@)/pgi && $(MAKE)

%.pgi.clean:
	$(SILENT)cd $(subst .pgi.clean,,$@)/pgi && $(MAKE) clean

%.ppcg:
	$(SILENT)cd $(subst .ppcg,,$@)/ppcg && $(MAKE)

%.ppcg.clean:
	$(SILENT)cd $(subst .ppcg.clean,,$@)/ppcg && $(MAKE) clean

%.cuda:
	$(SILENT)cd $(subst .cuda,,$@)/cuda && $(MAKE)

%.cuda.check:
	$(SILENT)cd $(subst .cuda.check,,$@)/cuda && $(MAKE) check

%.cuda.clean:
	$(SILENT)cd $(subst .cuda.clean,,$@)/cuda && $(MAKE) clean

%.cuda_shfl:
	$(SILENT)cd $(subst .cuda_shfl,,$@)/cuda_shfl && $(MAKE)

%.cuda_shfl.check:
	$(SILENT)cd $(subst .cuda_shfl.check,,$@)/cuda_shfl && $(MAKE) check

%.cuda_shfl.clean:
	$(SILENT)cd $(subst .cuda_shfl.clean,,$@)/cuda_shfl && $(MAKE) clean

%.cuda_shmem1d:
	$(SILENT)cd $(subst .cuda_shmem1d,,$@)/cuda_shmem1d && $(MAKE)

%.cuda_shmem1d.check:
	$(SILENT)cd $(subst .cuda_shmem1d.check,,$@)/cuda_shmem1d && $(MAKE) check

%.cuda_shmem1d.clean:
	$(SILENT)cd $(subst .cuda_shmem1d.clean,,$@)/cuda_shmem1d && $(MAKE) clean

%.cuda_shmem1dreg1d:
	$(SILENT)cd $(subst .cuda_shmem1dreg1d,,$@)/cuda_shmem1dreg1d && $(MAKE)

%.cuda_shmem1dreg1d.check:
	$(SILENT)cd $(subst .cuda_shmem1dreg1d.check,,$@)/cuda_shmem1dreg1d && $(MAKE) check

%.cuda_shmem1dreg1d.clean:
	$(SILENT)cd $(subst .cuda_shmem1dreg1d.clean,,$@)/cuda_shmem1dreg1d && $(MAKE) clean

%.cuda_shmem1dreg1dvec2:
	$(SILENT)cd $(subst .cuda_shmem1dreg1dvec2,,$@)/cuda_shmem1dreg1dvec2 && $(MAKE)

%.cuda_shmem1dreg1dvec2.check:
	$(SILENT)cd $(subst .cuda_shmem1dreg1dvec2.check,,$@)/cuda_shmem1dreg1dvec2 && $(MAKE) check

%.cuda_shmem1dreg1dvec2.clean:
	$(SILENT)cd $(subst .cuda_shmem1dreg1dvec2.clean,,$@)/cuda_shmem1dreg1dvec2 && $(MAKE) clean

%.cuda_shmem1dvec2:
	$(SILENT)cd $(subst .cuda_shmem1dvec2,,$@)/cuda_shmem1dvec2 && $(MAKE)

%.cuda_shmem1dvec2.check:
	$(SILENT)cd $(subst .cuda_shmem1dvec2.check,,$@)/cuda_shmem1dvec2 && $(MAKE) check

%.cuda_shmem1dvec2.clean:
	$(SILENT)cd $(subst .cuda_shmem1dvec2.clean,,$@)/cuda_shmem1dvec2 && $(MAKE) clean

%.cuda_shmem2d:
	$(SILENT)cd $(subst .cuda_shmem2d,,$@)/cuda_shmem2d && $(MAKE)

%.cuda_shmem2d.check:
	$(SILENT)cd $(subst .cuda_shmem2d.check,,$@)/cuda_shmem2d && $(MAKE) check

%.cuda_shmem2d.clean:
	$(SILENT)cd $(subst .cuda_shmem2d.clean,,$@)/cuda_shmem2d && $(MAKE) clean

%.cuda_shmem2dreg1d:
	$(SILENT)cd $(subst .cuda_shmem2dreg1d,,$@)/cuda_shmem2dreg1d && $(MAKE)

%.cuda_shmem2dreg1d.check:
	$(SILENT)cd $(subst .cuda_shmem2dreg1d.check,,$@)/cuda_shmem2dreg1d && $(MAKE) check

%.cuda_shmem2dreg1d.clean:
	$(SILENT)cd $(subst .cuda_shmem2dreg1d.clean,,$@)/cuda_shmem2dreg1d && $(MAKE) clean

%.cuda_shmem2dreg1dvec2:
	$(SILENT)cd $(subst .cuda_shmem2dreg1dvec2,,$@)/cuda_shmem2dreg1dvec2 && $(MAKE)

%.cuda_shmem2dreg1dvec2.check:
	$(SILENT)cd $(subst .cuda_shmem2dreg1dvec2.check,,$@)/cuda_shmem2dreg1dvec2 && $(MAKE) check

%.cuda_shmem2dreg1dvec2.clean:
	$(SILENT)cd $(subst .cuda_shmem2dreg1dvec2.clean,,$@)/cuda_shmem2dreg1dvec2 && $(MAKE) clean

%.cuda_shmem2dvec2:
	$(SILENT)cd $(subst .cuda_shmem2dvec2,,$@)/cuda_shmem2dvec2 && $(MAKE)

%.cuda_shmem2dvec2.check:
	$(SILENT)cd $(subst .cuda_shmem2dvec2.check,,$@)/cuda_shmem2dvec2 && $(MAKE) check

%.cuda_shmem2dvec2.clean:
	$(SILENT)cd $(subst .cuda_shmem2dvec2.clean,,$@)/cuda_shmem2dvec2 && $(MAKE) clean

%.cuda_vec2:
	$(SILENT)cd $(subst .cuda_vec2,,$@)/cuda_vec2 && $(MAKE)

%.cuda_vec2.check:
	$(SILENT)cd $(subst .cuda_vec2.check,,$@)/cuda_vec2 && $(MAKE) check

%.cuda_vec2.clean:
	$(SILENT)cd $(subst .cuda_vec2.clean,,$@)/cuda_vec2 && $(MAKE) clean

%.cuda_vec2shfl:
	$(SILENT)cd $(subst .cuda_vec2shfl,,$@)/cuda_vec2shfl && $(MAKE)

%.cuda_vec2shfl.check:
	$(SILENT)cd $(subst .cuda_vec2shfl.check,,$@)/cuda_vec2shfl && $(MAKE) check

%.cuda_vec2shfl.clean:
	$(SILENT)cd $(subst .cuda_vec2shfl.clean,,$@)/cuda_vec2shfl && $(MAKE) clean

test.gcc: $(TARGETS_GCC)
	$(SILENT)./benchmark $(NX) $(NY) $(NS) $(NT) $(NRUNS) gcc

test.kernelgen: $(TARGETS_KERNELGEN)
	$(SILENT)kernelgen_runmode=1 kernelgen_szheap=$$((1024*1024*800)) kernelgen_verbose=$$((1<<6)) ./benchmark $(NX) $(NY) $(NS) $(NT) $(NRUNS) kernelgen

test.pgi: $(TARGETS_PGI)
	$(SILENT)./benchmark $(NX) $(NY) $(NS) $(NT) $(NRUNS) pgi

test.ppcg: $(TARGETS_PPCG)
	$(SILENT)./benchmark $(NX) $(NY) $(NS) $(NT) $(NRUNS) ppcg

test.cuda: $(TARGETS_CAPS)
	$(SILENT)./benchmark $(NX) $(NY) $(NS) $(NT) $(NRUNS) cuda

test.cuda_shfl: $(TARGETS_CUDA_SHFL)
	$(SILENT)./benchmark $(NX) $(NY) $(NS) $(NT) $(NRUNS) cuda_shfl

test.cuda_shmem1d: $(TARGETS_CUDA_SHMEM1D)
	$(SILENT)./benchmark $(NX) $(NY) $(NS) $(NT) $(NRUNS) cuda_shmem1d

test.cuda_shmem1dreg1d: $(TARGETS_CUDA_SHMEM1DREG1D)
	$(SILENT)./benchmark $(NX) $(NY) $(NS) $(NT) $(NRUNS) cuda_shmem1dreg1d

test.cuda_shmem1dreg1dvec2: $(TARGETS_CUDA_SHMEM1DREG1DVEC2)
	$(SILENT)./benchmark $(NX) $(NY) $(NS) $(NT) $(NRUNS) cuda_shmem1dreg1dvec2

test.cuda_shmem1dvec2: $(TARGETS_CUDA_SHMEM1DVEC2)
	$(SILENT)./benchmark $(NX) $(NY) $(NS) $(NT) $(NRUNS) cuda_shmem1dvec2

test.cuda_shmem2d: $(TARGETS_CUDA_SHMEM2D)
	$(SILENT)./benchmark $(NX) $(NY) $(NS) $(NT) $(NRUNS) cuda_shmem2d

test.cuda_shmem2dreg1d: $(TARGETS_CUDA_SHMEM2DREG1D)
	$(SILENT)./benchmark $(NX) $(NY) $(NS) $(NT) $(NRUNS) cuda_shmem2dreg1d

test.cuda_shmem2dreg1dvec2: $(TARGETS_CUDA_SHMEM2DREG1DVEC2)
	$(SILENT)./benchmark $(NX) $(NY) $(NS) $(NT) $(NRUNS) cuda_shmem2dreg1dvec2

test.cuda_shmem2dvec2: $(TARGETS_CUDA_SHMEM2DVEC2)
	$(SILENT)./benchmark $(NX) $(NY) $(NS) $(NT) $(NRUNS) cuda_shmem2dvec2

test.cuda_vec2: $(TARGETS_CUDA_VEC2)
	$(SILENT)./benchmark $(NX) $(NY) $(NS) $(NT) $(NRUNS) cuda_vec2

test.cuda_vec2shfl: $(TARGETS_CUDA_VEC2SHFL)
	$(SILENT)./benchmark $(NX) $(NY) $(NS) $(NT) $(NRUNS) cuda_vec2shfl

test.cuda_all: $(TARGETS_CUDA) $(TARGETS_CUDA_SHFL) $(TARGETS_CUDA_SHMEM1D) \
	$(TARGETS_CUDA_SHMEM1DREG1D) $(TARGETS_CUDA_SHMEM1DREG1DVEC2) \
	$(TARGETS_CUDA_SHMEM1DVEC2) $(TARGETS_CUDA_SHMEM2D) $(TARGETS_CUDA_SHMEM2DREG1D) \
	$(TARGETS_CUDA_SHMEM2DREG1DVEC2) $(TARGETS_CUDA_SHMEM2DVEC2) $(TARGETS_CUDA_VEC2) \
	$(TARGETS_CUDA_VEC2SHFL)
	$(SILENT)./benchmark $(NX) $(NY) $(NS) $(NT) $(NRUNS) cuda cuda_shfl cuda_shmem1d cuda_shmem1dreg1d cuda_shmem1dreg1dvec2 cuda_shmem1dvec2 cuda_shmem2d cuda_shmem2dreg1d cuda_shmem2dreg1dvec2 cuda_shmem2dvec2 cuda_vec2 cuda_vec2shfl

