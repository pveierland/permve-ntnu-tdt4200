#include <cuda_runtime.h>

#include <cstddef>
#include <cstdio>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <sys/time.h>
#include <vector>

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

    for (int i = 0; i < 100; ++i)
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
    for (int i = 0; i < 8; ++i)
    {
        if (threadIdx.x % 8 == i || threadIdx.y % 8 == i)
        {
            some_other_task();
        }
    }
}

__global__
void
kernel2()
{
    for (int i = 0; i < 8; ++i)
    {
        if (blockIdx.x % 8 == i || blockIdx.y % 8 == i)
        {
            some_other_task();
        }
    }
}

cudaEvent_t e1, e2;

template <typename Func>
float
benchmark(Func f, dim3 grid_size, dim3 block_size, const int iterations)
{
    std::vector<float> results;
    results.reserve(iterations);

    for (int i = 0; i != iterations; ++i)
    {
        cuda_call(cudaEventRecord, e1);
        cuda_launch(f, grid_size, block_size);
        cuda_call(cudaEventRecord, e2);
        cuda_call(cudaEventSynchronize, e2);

        float p;
        cuda_call(cudaEventElapsedTime, &p, e1, e2);

        results.push_back(p);
    }

    return std::accumulate(
        results.begin(), results.end(), 0.0) / results.size();
}

int
main(const int argc, const char* argv[])
{
    try
    {
        cuda_call(cudaEventCreate, &e1);
        cuda_call(cudaEventCreate, &e2);
        cuda_call(cudaFree, 0);

        const dim3 grid_size(32, 32);
        const dim3 block_size(32, 32);

        const float k1 = benchmark(kernel1, grid_size, block_size, 5);
        printf("\n\n\n\n");
        const float k2 = benchmark(kernel2, grid_size, block_size, 5);

        // Convert output to microseconds:
        std::printf("%f\n%f\n", k1 * 1000.0f, k2 * 1000.0f);
    }
    catch (std::runtime_error& error)
    {
        std::cerr << error.what() << std::endl;
        return 1;
    }
}

