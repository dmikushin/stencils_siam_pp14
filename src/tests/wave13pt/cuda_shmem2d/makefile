include ../cuda/makefile

NVCC += -D__CUDA_SHMEM2D__

ifeq ($(real), float)
NVCC += --maxrregcount=32
endif
