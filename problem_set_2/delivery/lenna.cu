#include "lodepng.h"
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdio>
#include <iostream>
#include <stdexcept>
#include <string>

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
        image lenna("lenna512x512_inv.png");

        void* device_image_area;
        cuda_call(cudaMalloc, &device_image_area, lenna.data.size());

        cuda_call(cudaMemcpy,
                  device_image_area, &lenna.data.front(),
                  lenna.data.size(), cudaMemcpyHostToDevice);

        cuda_launch(invert_pixels, 384, 256, device_image_area);

        cuda_call(cudaMemcpy,
                  &lenna.data.front(), device_image_area,
                  lenna.data.size(), cudaMemcpyDeviceToHost);
        
        cuda_call(cudaFree, device_image_area);

        lenna.save("lenna512x512_orig.png");
    }
    catch (std::runtime_error& error)
    {
        std::cerr << error.what() << std::endl;
        return 1;
    }
}

