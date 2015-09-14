#include "lodepng.h"
#include <cuda_runtime.h>

#include <cstddef>
#include <iostream>
#include <stdexcept>
#include <string>
#include <cstdio>

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
//    std::cout << "  cacheModeCA        : " << attr.cacheModeCA        << std::endl;
    std::cout << "  constSizeBytes     : " << attr.constSizeBytes     << std::endl;
    std::cout << "  localSizeBytes     : " << attr.localSizeBytes     << std::endl;
    std::cout << "  maxThreadsPerBlock : " << attr.maxThreadsPerBlock << std::endl;
    std::cout << "  numRegs            : " << attr.numRegs            << std::endl;
    std::cout << "  ptxVersion         : " << attr.ptxVersion         << std::endl;
    std::cout << "  sharedSizeBytes    : " << attr.sharedSizeBytes    << std::endl;
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
    unsigned char* data = static_cast<unsigned char*>(image);
    int index = 3 * (blockIdx.y * width + blockIdx.x) + blockIdx.z;
    data[index] = ~data[index];
}

void DisplayHeader()
{
    const int kb = 1024;
    const int mb = kb * kb;
    std::cout << "NBody.GPU" << std::endl << "=========" << std::endl << std::endl;
    
    std::cout << "CUDA version:   v" << CUDART_VERSION << std::endl;    
    
    int devCount;
    cudaGetDeviceCount(&devCount);
    std::cout << "CUDA Devices: " << std::endl << std::endl;
    
    for (int i = 0; i < devCount; ++i)
    {
        cudaDeviceProp props;
        cudaGetDeviceProperties(&props, i);
        std::cout << i << ": " << props.name << ": " << props.major << "." << props.minor << std::endl;
        std::cout << "  Global memory:   " << props.totalGlobalMem / mb << "mb" << std::endl;
        std::cout << "  Shared memory:   " << props.sharedMemPerBlock / kb << "kb" << std::endl;
        std::cout << "  Constant memory: " << props.totalConstMem / kb << "kb" << std::endl;
        std::cout << "  Block registers: " << props.regsPerBlock << std::endl << std::endl;
        
        std::cout << "  Warp size:         " << props.warpSize << std::endl;
        std::cout << "  Threads per block: " << props.maxThreadsPerBlock << std::endl;
        std::cout << "  Max block dimensions: [ " << props.maxThreadsDim[0] << ", " << props.maxThreadsDim[1]  << ", " << props.maxThreadsDim[2] << " ]" << std::endl;
        std::cout << "  Max grid dimensions:  [ " << props.maxGridSize[0] << ", " << props.maxGridSize[1]  << ", " << props.maxGridSize[2] << " ]" << std::endl;
        std::cout << "  Number of multiprocessors: " << props.multiProcessorCount << std::endl;
        std::cout << "  Number of concurrent kernels: " << props.concurrentKernels << std::endl;
        std::cout << "  Maximum number of resident threads per multiprocessor: " << props.maxThreadsPerMultiProcessor << std::endl;
        std::cout << "  Memory bus width: " << props.memoryBusWidth << std::endl;
        std::cout << std::endl;
    }
}

int
main(const int argc, const char* argv[])
{
    cuda_print_function_attributes_(invert_pixel);
    return 0;
    DisplayHeader();
    return 0;
    try
    {
        image lenna("lenna512x512_inv.png");

        void* device_input;
        cuda_call(cudaMalloc, &device_input, lenna.data.size());

        cuda_call(cudaMemcpy,
                  device_input, &lenna.data.front(),
                  lenna.data.size(), cudaMemcpyHostToDevice);

        dim3 grid_dim(512, 512, 3);

        cuda_launch(invert_pixel, grid_dim, 1, device_input, lenna.width, lenna.height); 

        cuda_call(cudaMemcpy,
                  &lenna.data.front(), device_input,
                  lenna.data.size(), cudaMemcpyDeviceToHost);
        
        cuda_call(cudaFree, device_input);
        
        lenna.save("lenna512x512_orig.png");
    }
    catch (std::runtime_error& error)
    {
        std::cerr << error.what() << std::endl;
        return 1;
    }
}

