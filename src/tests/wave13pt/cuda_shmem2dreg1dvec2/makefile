include ../cuda/makefile

NVCC += -D__CUDA_SHMEM2D__ -D__CUDA_VECTORIZE2__ -D__CUDA_SHMEM2DREG1D__

ifeq ($(real), double)
NVCC += -Dreal2=double2
else
NVCC += -Dreal2=float2 -Dreal4=float4
endif
