#include "lodepng.h"
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdio>
#include <iostream>
#include <stdexcept>
#include <string>
#include <sys/time.h>

#define cuda_call(f, ...) \
    cuda_assert(f(__VA_ARGS__), __FILE__, __LINE__, #f)

#define cuda_launch(kernel, grid_dim, block_dim, ...) \
{ \
    kernel<<<grid_dim, block_dim>>>(__VA_ARGS__); \
    cuda_assert(cudaPeekAtLastError(), __FILE__, __LINE__, "kernel " #kernel " launch"); \
}

inline
void
cuda_assert(cudaError_t error, const char* file, const int line, const char* function)
{
    if (error)
    {
        throw std::runtime_error(std::string(function) + " failed: " + cudaGetErrorString(error));
    }
}

#define cuda_print_function_attributes_(f) \
    cuda_print_function_attributes(#f, (const void*)f)

__device__
void
some_other_task()
{
    volatile double x;

    for (int i = 0; i < 1000; ++i)
    {
        x = std::log(x) / 22.0;
        x *= 42.0;
        x /= std::sin(x);
    }
}

__global__
void
kernel1()
{
    some_other_task();
}

int
main(const int argc, const char* argv[])
{
    try
    {
        cudaEvent_t e1, e2;
        
        cuda_call(cudaEventCreate, &e1);
        cuda_call(cudaEventCreate, &e2);

        cuda_call(cudaFree, 0);

        cuda_call(cudaEventRecord, e1);
        cuda_launch(kernel1, 1, 1);
        cuda_call(cudaEventRecord, e2);

        cuda_call(cudaEventSynchronize, e2);

        float p;

        cuda_call(cudaEventElapsedTime, &p, e1, e2);

        // Convert output to microseconds:
        std::printf("%f\n", p * 1000.0f);
    }
    catch (std::runtime_error& error)
    {
        std::cerr << error.what() << std::endl;
        return 1;
    }
}

