.PHONY: clean all

MATH:=-lm
CFLAGS:=--std=c99
CUDAFLAGS:=--compiler-bindir g++-4.8 -arch=sm_20

cpu_version: cpu_version.o lodepng.c.o
	$(CC) -o $@ $^

%.cu.o: %.cu
	nvcc $(CUDAFLAGS) $< -o $@ -c

cuda: gpu_version

gpu_version: pinkfloyd.cu.o lodepng.cpp.o
	nvcc $(CUDAFLAGS) $^ -o $@

color.c.o: color.c
	$(CXX) -c $< -o $@

lodepng.c.o: lodepng.c
	$(CC) $(CFLAGS) -c $< -o $@

lodepng.cpp.o: lodepng.cpp
	$(CXX) -c $< -o $@

all: cpu_version gpu_version

cpu_version: cpu_version.o lodepng.o color.o
	$(CC) $^ -o $@ $(MATH)

clean:
	rm -f *.o cpu_version 
