include ../cuda/makefile

NVCC += -D__CUDA_VECTORIZE2__

ifeq ($(real), double)
NVCC += -Dreal2=double2
else
ifeq ($(GPUARCH), 35)
MAXRREGCOUNT = 80
endif
NVCC += -Dreal2=float2 -Dreal4=float4
endif
