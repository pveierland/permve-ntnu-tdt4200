.PHONY: clean all

CFLAGS:=--std=c99
CUDAFLAGS:=--compiler-bindir g++-4.8 -arch=sm_20

%.cu.o: %.cu
	nvcc $(CUDAFLAGS) $< -o $@ -c

cuda: gpu_version

gpu_version: full_warp.cu.o
	nvcc $(CUDAFLAGS) $^ -o $@

all: gpu_version

clean: 
	rm -f *.o gpu_version 

