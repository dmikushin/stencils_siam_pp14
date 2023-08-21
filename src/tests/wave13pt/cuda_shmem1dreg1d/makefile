include ../cuda/makefile

NVCC += -D__CUDA_SHMEM1D__ -D__CUDA_SHMEM1DREG1D__

ifeq ($(real), float)
NVCC += --maxrregcount=32
endif
