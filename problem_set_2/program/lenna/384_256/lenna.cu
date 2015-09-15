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
invert_pixels(void* image)
{
    unsigned long* data =
        static_cast<unsigned long*>(image) + (blockIdx.x << 8) + threadIdx.x;

    *data = ~*data;
}

int
main(const int argc, const char* argv[])
{
    try
    {
        const unsigned long t1 = get_wall_time();

        image lenna("lenna512x512_inv.png");

        const unsigned long t2 = get_wall_time();
        
        cudaEvent_t e1, e2, e3, e4, e5, e6;
        
        cuda_call(cudaEventCreate, &e1);
        cuda_call(cudaEventCreate, &e2);
        cuda_call(cudaEventCreate, &e3);
        cuda_call(cudaEventCreate, &e4);
        cuda_call(cudaEventCreate, &e5);
        cuda_call(cudaEventCreate, &e6);

        cuda_call(cudaFree, 0);

        const unsigned long t3 = get_wall_time();

        cuda_call(cudaEventRecord, e1);

        void* device_image_area;
        cuda_call(cudaMalloc, &device_image_area, lenna.data.size());

        cuda_call(cudaEventRecord, e2);

        cuda_call(cudaMemcpy,
                  device_image_area, &lenna.data.front(),
                  lenna.data.size(), cudaMemcpyHostToDevice);

        cuda_call(cudaEventRecord, e3);

        cuda_launch(invert_pixels, 384, 256, device_image_area);

        cuda_call(cudaEventRecord, e4);

        cuda_call(cudaMemcpy,
                  &lenna.data.front(), device_image_area,
                  lenna.data.size(), cudaMemcpyDeviceToHost);

        cuda_call(cudaEventRecord, e5);
        
        cuda_call(cudaFree, device_image_area);

        cuda_call(cudaEventRecord, e6);

        cuda_call(cudaEventSynchronize, e6);

        const unsigned long t4 = get_wall_time();
        
        lenna.save("lenna512x512_orig.png");

        const unsigned long t5 = get_wall_time();

        float p1, p2, p3, p4, p5;

        cuda_call(cudaEventElapsedTime, &p1, e1, e2);
        cuda_call(cudaEventElapsedTime, &p2, e2, e3);
        cuda_call(cudaEventElapsedTime, &p3, e3, e4);
        cuda_call(cudaEventElapsedTime, &p4, e4, e5);
        cuda_call(cudaEventElapsedTime, &p5, e5, e6);

        // Convert output to microseconds:
        std::printf("%lu,%lu,%lu,%lu,%f,%f,%f,%f,%f\n",
                    t5 - t1, t2 - t1, t3 - t2, t5 - t4,
                    p1 * 1000, p2 * 1000, p3 * 1000, p4 * 1000, p5 * 1000);
    }
    catch (std::runtime_error& error)
    {
        std::cerr << error.what() << std::endl;
        return 1;
    }
}

