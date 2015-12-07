#include <iostream>
#include <math.h>
#include <sstream>
#include <stdexcept>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>

#include "ppmCU.h"

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
        std::ostringstream oss;
        oss << "CUDA function "
            << function
            << " (" << file << ":" << line << ") "
            << "failed: " << cudaGetErrorString(error);
        throw std::runtime_error(oss.str());
    }
}

__global__
void
performNewIdeaIterationGPU(float* imageOut, float*  imageIn, int image_width, int image_height, int size)
{
    // For each pixel we compute the magic number
    float sumR = 0;
    float sumG = 0;
    float sumB = 0;
    int countIncluded = 0;

    int senterY = blockDim.y * blockIdx.y + threadIdx.y;
    int senterX = blockDim.x * blockIdx.x + threadIdx.x;

    for(int x = -size; x <= size; x++) {
        for(int y = -size; y <= size; y++) {
            int currentX = senterX + x;
            int currentY = senterY + y;

            // Check if we are outside the bounds
            if(currentX < 0)
                continue;
            if(currentX >= image_width)
                continue;
            if(currentY < 0)
                continue;
            if(currentY >= image_height)
                continue;

            // Now we can begin
            int offsetOfThePixel = 3 * (image_width * currentY + currentX);

            sumR += imageIn[offsetOfThePixel + 0];
            sumG += imageIn[offsetOfThePixel + 1];
            sumB += imageIn[offsetOfThePixel + 2];

            // Keep track of how many values we have included
            countIncluded++;
        }
    }

    // Now we compute the final value for all colours
    float valueR = sumR / countIncluded;
    float valueG = sumG / countIncluded;
    float valueB = sumB / countIncluded;

    // Update the output image
    int offsetOfThePixel = 3 * (image_width * senterY + senterX);
    imageOut[offsetOfThePixel + 0] = valueR;
    imageOut[offsetOfThePixel + 1] = valueG;
    imageOut[offsetOfThePixel + 2] = valueB;
}

__device__ inline
unsigned char
computeFinalPixelValue(float value)
{
    if (value <= -1.0f)
    {
        value += 257.0f;
    }
    return (unsigned char)(floorf(min(max(value, 0.0f), 255.0f)));
}

// performNewIdeaFinalizationGPU requires that image has width divisible by 4
__global__
void
performNewIdeaFinalizationGPU(
    const int            image_width,
    float* const         image_input_small,
    float* const         image_input_large,
    unsigned char* const image_output)
{
    const int offset = blockIdx.x * image_width * 3 /* values per pixel */
                     + threadIdx.x * 4 /* pixels per thread */ * 3 /* values per pixel */;

    for (int i = 0; i != 12; ++i)
    {
        image_output[offset + i] = computeFinalPixelValue(image_input_large[offset + i] - image_input_small[offset + i]);
    }
}

__global__
void
convertImageToNewFormatGPU(
    float* const         output_image,
    unsigned char* const input_image,
    const int            image_width)
{
    int offset = 3 * (image_width * (blockDim.y * blockIdx.y + threadIdx.y)
                                   + blockDim.x * blockIdx.x + threadIdx.x);

    output_image[offset + 0] = (float) input_image[offset + 0];
    output_image[offset + 1] = (float) input_image[offset + 1];
    output_image[offset + 2] = (float) input_image[offset + 2];
}

int
main(int argc, char** argv)
{
	PPMImage *image = readPPM("flower.ppm");

    const int pixels_in_image = image->x * image->y;
    const int bytes_per_pixel = 3 * sizeof(float);

    float*         device_image_unchanged = NULL;
    float*         device_image_buffer = NULL;
    float*         device_image_small = NULL;
    float*         device_image_big = NULL;
    unsigned char* device_image_x = NULL;

	PPMImage *imageOut;
	imageOut = (PPMImage *)malloc(sizeof(PPMImage));
    imageOut->x = image->x;
    imageOut->y = image->y;
	imageOut->data = (PPMPixel*)malloc(image->x * image->y * sizeof(PPMPixel));

    cuda_call(cudaMalloc, &device_image_unchanged, pixels_in_image * bytes_per_pixel);
    cuda_call(cudaMalloc, &device_image_buffer, pixels_in_image * bytes_per_pixel);
    cuda_call(cudaMalloc, &device_image_small, pixels_in_image * bytes_per_pixel);
    cuda_call(cudaMalloc, &device_image_big, pixels_in_image * bytes_per_pixel);
    cuda_call(cudaMalloc, &device_image_x, 3 * pixels_in_image);

    dim3 grid(30, 75);
    dim3 block(64, 16);

    cuda_call(cudaMemcpy, device_image_x, image->data,
              3 * pixels_in_image, cudaMemcpyHostToDevice);

    cuda_launch(convertImageToNewFormatGPU, grid, block, device_image_unchanged, device_image_x, image->x);

    cuda_launch(performNewIdeaIterationGPU, grid, block, device_image_small, device_image_unchanged, 1920, 1200, 2);
    cuda_launch(performNewIdeaIterationGPU, grid, block, device_image_buffer, device_image_small, 1920, 1200, 2);
    cuda_launch(performNewIdeaIterationGPU, grid, block, device_image_small, device_image_buffer, 1920, 1200, 2);
    cuda_launch(performNewIdeaIterationGPU, grid, block, device_image_buffer, device_image_small, 1920, 1200, 2);
    cuda_launch(performNewIdeaIterationGPU, grid, block, device_image_small, device_image_buffer, 1920, 1200, 2);

    cuda_launch(performNewIdeaIterationGPU, grid, block, device_image_big, device_image_unchanged, 1920, 1200, 3);
    cuda_launch(performNewIdeaIterationGPU, grid, block, device_image_buffer, device_image_big, 1920, 1200, 3);
    cuda_launch(performNewIdeaIterationGPU, grid, block, device_image_big, device_image_buffer, 1920, 1200, 3);
    cuda_launch(performNewIdeaIterationGPU, grid, block, device_image_buffer, device_image_big, 1920, 1200, 3);
    cuda_launch(performNewIdeaIterationGPU, grid, block, device_image_big, device_image_buffer, 1920, 1200, 3);

    cuda_launch(performNewIdeaFinalizationGPU, 1200, 480,
                1920, device_image_small, device_image_big, device_image_x);

    cuda_call(cudaMemcpy, imageOut->data, device_image_x,
              3 * pixels_in_image, cudaMemcpyDeviceToHost);

    writePPM("flower_tiny.ppm", imageOut);

    cuda_launch(performNewIdeaIterationGPU, grid, block, device_image_small, device_image_unchanged, 1920, 1200, 5);
    cuda_launch(performNewIdeaIterationGPU, grid, block, device_image_buffer, device_image_small, 1920, 1200, 5);
    cuda_launch(performNewIdeaIterationGPU, grid, block, device_image_small, device_image_buffer, 1920, 1200, 5);
    cuda_launch(performNewIdeaIterationGPU, grid, block, device_image_buffer, device_image_small, 1920, 1200, 5);
    cuda_launch(performNewIdeaIterationGPU, grid, block, device_image_small, device_image_buffer, 1920, 1200, 5);

    cuda_launch(performNewIdeaFinalizationGPU, 1200, 480,
                1920, device_image_big, device_image_small, device_image_x);

    cuda_call(cudaMemcpy, imageOut->data, device_image_x,
              3 * pixels_in_image, cudaMemcpyDeviceToHost);

    writePPM("flower_small.ppm", imageOut);

    cuda_launch(performNewIdeaIterationGPU, grid, block, device_image_big, device_image_unchanged, 1920, 1200, 8);
    cuda_launch(performNewIdeaIterationGPU, grid, block, device_image_buffer, device_image_big, 1920, 1200, 8);
    cuda_launch(performNewIdeaIterationGPU, grid, block, device_image_big, device_image_buffer, 1920, 1200, 8);
    cuda_launch(performNewIdeaIterationGPU, grid, block, device_image_buffer, device_image_big, 1920, 1200, 8);
    cuda_launch(performNewIdeaIterationGPU, grid, block, device_image_big, device_image_buffer, 1920, 1200, 8);

    cuda_launch(performNewIdeaFinalizationGPU, 1200, 480,
                1920, device_image_small, device_image_big, device_image_x);

    cuda_call(cudaMemcpy, imageOut->data, device_image_x,
              3 * pixels_in_image, cudaMemcpyDeviceToHost);

    writePPM("flower_medium.ppm", imageOut);

    cuda_call(cudaFree, device_image_unchanged);
    cuda_call(cudaFree, device_image_buffer);
    cuda_call(cudaFree, device_image_small);
    cuda_call(cudaFree, device_image_big);
    cuda_call(cudaFree, device_image_x);

	// free all memory structures
	free(imageOut->data);
	free(imageOut);
	free(image->data);
	free(image);

	return 0;
}

