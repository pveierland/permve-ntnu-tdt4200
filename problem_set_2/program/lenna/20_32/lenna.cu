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
    cuda_assert(cudaDeviceSynchronize(), __FILE__, __LINE__, "kernel " #kernel " synchronize"); \
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

void
cuda_print_function_attributes(const char* func_name, const void* func)
{
    cudaFuncAttributes attr;
    cuda_call(cudaFuncGetAttributes, &attr, func);

    std::cout << func_name << " cudaFuncAttributes"                   << std::endl;
    std::cout << "  binaryVersion      : " << attr.binaryVersion      << std::endl;
    std::cout << "  constSizeBytes     : " << attr.constSizeBytes     << std::endl;
    std::cout << "  localSizeBytes     : " << attr.localSizeBytes     << std::endl;
    std::cout << "  maxThreadsPerBlock : " << attr.maxThreadsPerBlock << std::endl;
    std::cout << "  numRegs            : " << attr.numRegs            << std::endl;
    std::cout << "  ptxVersion         : " << attr.ptxVersion         << std::endl;
    std::cout << "  sharedSizeBytes    : " << attr.sharedSizeBytes    << std::endl;
}

unsigned long
get_wall_time()
{
    struct timeval time;

    if (gettimeofday(&time, NULL))
    {
        throw std::runtime_error("gettimeofday failed");
        return 0;
    }

    return static_cast<unsigned long>(time.tv_sec) * 1000000 +
           static_cast<unsigned long>(time.tv_usec);
}

struct image
{
    image() { }

    image(const std::string& filename)
    {
        const unsigned int status = lodepng::decode(
            data, width, height, filename, LCT_RGB);

        if (status)
        {
            throw std::runtime_error(
                "failed to load image " + filename + ": " +
                lodepng_error_text(status));
        }
    }

    void
    save(const std::string filename)
    {
        const unsigned int status = lodepng::encode(
            filename, data, width, height, LCT_RGB);

        if (status)
        {
            throw std::runtime_error(
                "failed to save image " + filename + ": " +
                lodepng_error_text(status));
        }
    }

    unsigned int width, height;
    std::vector<unsigned char> data;
};

__global__
void
invert_pixel(void* image, int width, int height)
{
    unsigned long* data = static_cast<unsigned long*>(image) + blockIdx.x * 32 + threadIdx.x;

    for (int i = 153; i != 0; --i, data += 640)
    {
        *data = ~*data;
    }

    if (blockIdx.x < 12)
    {
        *data = ~*data;
    }
}

int
main(const int argc, const char* argv[])
{
    unsigned long t1, t2, t3, t4, t5, t6;

    try
    {
        image lenna("lenna512x512_inv.png");

        cuda_call(cudaFree, 0);

        t1 = get_wall_time();

        void* device_image_area;
        cuda_call(cudaMalloc, &device_image_area, lenna.data.size());
        
        t2 = get_wall_time();

        cuda_call(cudaMemcpy,
                  device_image_area, &lenna.data.front(),
                  lenna.data.size(), cudaMemcpyHostToDevice);

        t3 = get_wall_time();

        cuda_launch(invert_pixel, 20, 32, device_image_area, lenna.width, lenna.height); 

        t4 = get_wall_time();

        cuda_call(cudaMemcpy,
                  &lenna.data.front(), device_image_area,
                  lenna.data.size(), cudaMemcpyDeviceToHost);

        t5 = get_wall_time();
        
        cuda_call(cudaFree, device_image_area);

        t6 = get_wall_time();
        
        lenna.save("lenna512x512_orig.png");
        
        std::printf("%lu,%lu,%lu,%lu,%lu\n",
                    (t2 - t1), (t3 - t2), (t4 - t3), (t5 - t4), (t6 - t5));
    }
    catch (std::runtime_error& error)
    {
        std::cerr << error.what() << std::endl;
        return 1;
    }
}

