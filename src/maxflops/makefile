all: maxflops

maxflops: maxflops.cu
	nvcc -arch=sm_30 $< -o $@

clean:
	rm -rf maxflops
