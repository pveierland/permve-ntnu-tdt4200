#include <stdexcept>
#include <string>
#include <iostream>
#include <sstream>

#include <stdint.h>

#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
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

// Image from:
// http://7-themes.com/6971875-funny-flowers-pictures.html

// TODO: You must implement this
// The handout code is much simpler than the MPI/OpenMP versions
//__global__ void performNewIdeaIterationGPU( ... ) { ... }

// TODO: You should implement this
//__global__ void performNewIdeaFinalizationGPU( ... ) { ... }

/*
__device__ inline
unsigned char
computeFinalPixelValue(float value)
{
    if(value > 255.0f)
        return 255;
    else if (value < -1.0f) {
        value += 257.0f;

    if (value > 255.0f)
        return 255;
    else
        return (unsigned char)floorf(value);
    } else if (value > -1.0f && value < 0.0f) {
        return 0;
    } else {
        return (unsigned char)floorf(value);
    }
}
*/


template <int tile_size, int filter_size>
__global__
void
performNewIdeaIterationGPU(float* const image, const int image_width)
{
    __shared__ float s[tile_size + 2 * filter_size][3];

    const int column_index          = tile_size * blockIdx.x - filter_size + threadIdx.x;
    const int row_index             = image_width * blockIdx.y;
    const int is_valid_column_index = column_index >= 0 && column_index < image_width;

//    printf("column_index = %3d row_index = %3d is_valid_column_index = %d\n", column_index, row_index, is_valid_column_index);

    // Load pixel value into shared memory
    const int pixel_offset = 3 * (row_index + column_index);
    s[threadIdx.x][0] = is_valid_column_index ? image[pixel_offset + 0] : 0;
    s[threadIdx.x][1] = is_valid_column_index ? image[pixel_offset + 1] : 0;
    s[threadIdx.x][2] = is_valid_column_index ? image[pixel_offset + 2] : 0;

    __syncthreads();

    // Do filtering

    if (threadIdx.x >= filter_size && threadIdx.x < blockDim.x - filter_size)
    {
        //int start = max(column_index - filter_size, 0);
        //int end   = min(column_index + filter_size, image_width - 1);

        //float sum[3] = { 0 };

        //for (int i = start; i <= end; ++i)
        //{
        //    sum[0] += s[i][0];
        //    sum[1] += s[i][1];
        //    sum[2] += s[i][2];
        //}

        const int num_filtered_pixels = end - start + 1;

        //image[pixel_offset + 0] = sum[0] / num_filtered_pixels;
        //image[pixel_offset + 1] = sum[1] / num_filtered_pixels;
        //image[pixel_offset + 2] = sum[2] / num_filtered_pixels;

        image[pixel_offset + 0] = 0.0f;
        image[pixel_offset + 1] = 128.0f;
        image[pixel_offset + 2] = 255.0f;
    }
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
    const int      image_width,
    float* const   image_input_small,
    float* const   image_input_large,
    uint8_t* const image_output)
{
    const int offset = blockIdx.x * image_width * 3 /* values per pixel */
                     + threadIdx.x * 4 /* pixels per thread */ * 3 /* values per pixel */;

    for (int i = 0; i != 12; ++i)
    {
        image_output[offset + i] = computeFinalPixelValue(image_input_small[offset + i]);
        //image_output[offset + i] = computeFinalPixelValue(image_input_large[offset + i] - image_input_small[offset + i]);
    }
}

typedef struct {
     float red,green,blue;
} AccuratePixel;

typedef struct {
     int x, y;
     AccuratePixel *data;
} AccurateImage;

// TODO: You should implement this
//__global__ void convertImageToNewFormatGPU( ... ) { ... }

// Perhaps some extra kernels will be practical as well?
//__global__ void ...GPU( ... ) { ... }


// Convert a PPM image to a high-precision format
AccurateImage *convertImageToNewFormat(PPMImage *image) {
	// Make a copy
	AccurateImage *imageAccurate;
	imageAccurate = (AccurateImage *)malloc(sizeof(AccurateImage));
	imageAccurate->data = (AccuratePixel*)malloc(image->x * image->y * sizeof(AccuratePixel));
	for(int i = 0; i < image->x * image->y; i++) {
		imageAccurate->data[i].red   = (float) image->data[i].red;
		imageAccurate->data[i].green = (float) image->data[i].green;
		imageAccurate->data[i].blue  = (float) image->data[i].blue;
	}
	imageAccurate->x = image->x;
	imageAccurate->y = image->y;

	return imageAccurate;
}

// Convert a high-precision format to a PPM image
PPMImage *convertNewFormatToPPM(AccurateImage *image) {
	// Make a copy
	PPMImage *imagePPM;
	imagePPM = (PPMImage *)malloc(sizeof(PPMImage));
	imagePPM->data = (PPMPixel*)malloc(image->x * image->y * sizeof(PPMPixel));
	for(int i = 0; i < image->x * image->y; i++) {
		imagePPM->data[i].red   = (unsigned char) image->data[i].red;
		imagePPM->data[i].green = (unsigned char) image->data[i].green;
		imagePPM->data[i].blue  = (unsigned char) image->data[i].blue;
	}
	imagePPM->x = image->x;
	imagePPM->y = image->y;

	return imagePPM;
}

AccurateImage *createEmptyImage(PPMImage *image){
	AccurateImage *imageAccurate;
	imageAccurate = (AccurateImage *)malloc(sizeof(AccurateImage));
	imageAccurate->data = (AccuratePixel*)malloc(image->x * image->y * sizeof(AccuratePixel));
	imageAccurate->x = image->x;
	imageAccurate->y = image->y;

	return imageAccurate;
}

// free memory of an AccurateImage
void freeImage(AccurateImage *image){
	free(image->data);
	free(image);
}

void performNewIdeaIteration(AccurateImage *imageOut, AccurateImage *imageIn, int size) {

	// Iterate over each pixel
	for(int senterX = 0; senterX < imageIn->x; senterX++) {

		for(int senterY = 0; senterY < imageIn->y; senterY++) {

			// For each pixel we compute the magic number
			float sumR = 0;
			float sumG = 0;
			float sumB = 0;
			int countIncluded = 0;
			for(int x = -size; x <= size; x++) {

				for(int y = -size; y <= size; y++) {
					int currentX = senterX + x;
					int currentY = senterY + y;

					// Check if we are outside the bounds
					if(currentX < 0)
						continue;
					if(currentX >= imageIn->x)
						continue;
					if(currentY < 0)
						continue;
					if(currentY >= imageIn->y)
						continue;

					// Now we can begin
					int numberOfValuesInEachRow = imageIn->x;
					int offsetOfThePixel = (numberOfValuesInEachRow * currentY + currentX);
					sumR += imageIn->data[offsetOfThePixel].red;
					sumG += imageIn->data[offsetOfThePixel].green;
					sumB += imageIn->data[offsetOfThePixel].blue;

					// Keep track of how many values we have included
					countIncluded++;
				}

			}

			// Now we compute the final value for all colours
			float valueR = sumR / countIncluded;
			float valueG = sumG / countIncluded;
			float valueB = sumB / countIncluded;

			// Update the output image
			int numberOfValuesInEachRow = imageOut->x; // R, G and B
			int offsetOfThePixel = (numberOfValuesInEachRow * senterY + senterX);
			imageOut->data[offsetOfThePixel].red = valueR;
			imageOut->data[offsetOfThePixel].green = valueG;
			imageOut->data[offsetOfThePixel].blue = valueB;
		}
	}
}

//inline int
//computeFinalPixelValue(float value)
//{
//    if (value <= -1.0f)
//    {
//        value += 257.0f;
//    }
//    return (int)(floorf(min(max(value, 0.0f), 255.0f)));
//}
//
//void performNewIdeaFinalization(AccurateImage *imageInSmall, AccurateImage *imageInLarge, PPMImage *imageOut) {
//
//
//	imageOut->x = imageInSmall->x;
//	imageOut->y = imageInSmall->y;
//
//	for(int i = 0; i < imageInSmall->x * imageInSmall->y; i++) {
//        imageOut->data[i].red   = computeFinalPixelValue(imageInLarge->data[i].red - imageInSmall->data[i].red);
//        imageOut->data[i].green = computeFinalPixelValue(imageInLarge->data[i].green - imageInSmall->data[i].green);
//        imageOut->data[i].blue  = computeFinalPixelValue(imageInLarge->data[i].blue - imageInSmall->data[i].blue);
//	}
//}

//// Perform the final step, and save it as a ppm in imageOut
//void performNewIdeaFinalization(AccurateImage *imageInSmall, AccurateImage *imageInLarge, PPMImage *imageOut) {
//
//
//	imageOut->x = imageInSmall->x;
//	imageOut->y = imageInSmall->y;
//
//	for(int i = 0; i < imageInSmall->x * imageInSmall->y; i++) {
//		float value = (imageInLarge->data[i].red - imageInSmall->data[i].red);
//		if(value > 255.0f)
//			imageOut->data[i].red = 255;
//		else if (value < -1.0f) {
//			value = 257.0f+value;
//			if(value > 255.0f)
//				imageOut->data[i].red = 255;
//			else
//				imageOut->data[i].red = floorf(value);
//		} else if (value > -1.0f && value < 0.0f) {
//			imageOut->data[i].red = 0;
//		} else {
//			imageOut->data[i].red = floorf(value);
//		}
//
//		value = (imageInLarge->data[i].green - imageInSmall->data[i].green);
//		if(value > 255.0f)
//			imageOut->data[i].green = 255;
//		else if (value < -1.0f) {
//			value = 257.0f+value;
//			if(value > 255.0f)
//				imageOut->data[i].green = 255;
//			else
//				imageOut->data[i].green = floorf(value);
//		} else if (value > -1.0f && value < 0.0f) {
//			imageOut->data[i].green = 0;
//		} else {
//			imageOut->data[i].green = floorf(value);
//		}
//
//		value = (imageInLarge->data[i].blue - imageInSmall->data[i].blue);
//		if(value > 255.0f)
//			imageOut->data[i].blue = 255;
//		else if (value < -1.0f) {
//			value = 257.0f+value;
//			if(value > 255.0f)
//				imageOut->data[i].blue = 255;
//			else
//				imageOut->data[i].blue = floorf(value);
//		} else if (value > -1.0f && value < 0.0f) {
//			imageOut->data[i].blue = 0;
//		} else {
//			imageOut->data[i].blue = floorf(value);
//		}
//	}
//}

// 1920 x 1200
int main(int argc, char** argv) {
	PPMImage *image;

	if(argc > 1) {
		image = readPPM("flower.ppm");
	} else {
		image = readStreamPPM(stdin);
	}

    std::cout << "P" << std::endl;

    const int pixels_in_image = image->x * image->y;
    const int bytes_per_pixel = 3 * sizeof(float);

    float*         device_image_a      = NULL;
    float*         device_image_b      = NULL;
    unsigned char* device_image_output = NULL;

    cuda_call(cudaMalloc, &device_image_a, pixels_in_image * bytes_per_pixel);
    cuda_call(cudaMalloc, &device_image_b, pixels_in_image * bytes_per_pixel);
    cuda_call(cudaMalloc, &device_image_output, 3 * pixels_in_image);

	AccurateImage *imageUnchanged = convertImageToNewFormat(image); // save the unchanged image from input image
	AccurateImage *imageBuffer = createEmptyImage(image);
	AccurateImage *imageSmall = createEmptyImage(image);
	AccurateImage *imageBig = createEmptyImage(image);

	PPMImage *imageOut;
	imageOut = (PPMImage *)malloc(sizeof(PPMImage));
    imageOut->x = image->x;
    imageOut->y = image->y;
	imageOut->data = (PPMPixel*)malloc(image->x * image->y * sizeof(PPMPixel));

    std::cout << "E" << std::endl;

    cuda_call(cudaMemcpy, device_image_a, imageUnchanged->data,
              pixels_in_image * bytes_per_pixel, cudaMemcpyHostToDevice);

    #define TILE_WIDTH 30
    dim3 grid_size(1920 / TILE_WIDTH, 1200);
    dim3 block_size(TILE_WIDTH + 2 * 5);

    #define performNewIdeaIterationGPU_template performNewIdeaIterationGPU<TILE_WIDTH, 5>

    cuda_launch(performNewIdeaIterationGPU_template, grid_size, block_size, device_image_a, 1920);

//	// Process the tiny case:
//	performNewIdeaIteration(imageSmall, imageUnchanged, 2);
//	performNewIdeaIteration(imageBuffer, imageSmall, 2);
//	performNewIdeaIteration(imageSmall, imageBuffer, 2);
//	performNewIdeaIteration(imageBuffer, imageSmall, 2);
//	performNewIdeaIteration(imageSmall, imageBuffer, 2);
//
//	// Process the small case:
//	performNewIdeaIteration(imageBig, imageUnchanged,3);
//	performNewIdeaIteration(imageBuffer, imageBig,3);
//	performNewIdeaIteration(imageBig, imageBuffer,3);
//	performNewIdeaIteration(imageBuffer, imageBig,3);
//	performNewIdeaIteration(imageBig, imageBuffer,3);

    std::cout << "N " << device_image_a << std::endl;


    std::cout << "X" << std::endl;

//    cuda_call(cudaMemcpy, device_image_b, imageBig->data,
//              pixels_in_image * bytes_per_pixel, cudaMemcpyHostToDevice);

    std::cout << "Y" << std::endl;

    cuda_launch(performNewIdeaFinalizationGPU, 1200, 480,
                1920, device_image_a, device_image_b, device_image_output);

    std::cout << "Z" << std::endl;

    cuda_call(cudaMemcpy,
              imageOut->data,
              device_image_output,
              3 * pixels_in_image,
              cudaMemcpyDeviceToHost);

    std::cout << "1" << std::endl;

    //PPMImage* fresh = performNewIdeaFinalization(imageSmall, imageBig);

    //for (int i = 0; i != pixels_in_image; ++i) {
    //    if (imageOut->data[i].red != fresh->data[i].red) {
    //        printf("NO MATCH %d", i);
    //        break;
    //    }
    //}

    //printf("\nHost output:\n");

    //for (int i = 0; i != 16; ++i)
    //{
    //    printf("0x%02X ", ((unsigned char*) imageOut->data)[i]);
    //}
    //printf("\n");

    //for (int i = 0; i != 16; ++i)
    //{
    //    printf("0x%02X ", ((unsigned char*) fresh->data)[i]);
    //}
    //printf("\n");

	if(argc > 1) {
		writePPM("flower_tiny.ppm", imageOut);
	} else {
		writeStreamPPM(stdout, imageOut);
	}

//	// Process the medium case:
//	performNewIdeaIteration(imageSmall, imageUnchanged, 5);
//	performNewIdeaIteration(imageBuffer, imageSmall, 5);
//	performNewIdeaIteration(imageSmall, imageBuffer, 5);
//	performNewIdeaIteration(imageBuffer, imageSmall, 5);
//	performNewIdeaIteration(imageSmall, imageBuffer, 5);
//
//	// save small case
//    cuda_call(cudaMemcpy, device_image_a, imageBig->data,
//              pixels_in_image * bytes_per_pixel, cudaMemcpyHostToDevice);
//
//    cuda_call(cudaMemcpy, device_image_b, imageSmall->data,
//              pixels_in_image * bytes_per_pixel, cudaMemcpyHostToDevice);
//
//    cuda_launch(performNewIdeaFinalizationGPU, 1200, 240,
//                1920, device_image_a, device_image_b, device_image_output);
//
//    cuda_call(cudaMemcpy,
//              imageOut->data,
//              device_image_output,
//              pixels_in_image,
//              cudaMemcpyDeviceToHost);
//	if(argc > 1) {
//		writePPM("flower_small.ppm", imageOut);
//	} else {
//		writeStreamPPM(stdout, imageOut);
//	}
//
//    std::cout << "5" << std::endl;
//
//	// process the large case
//	performNewIdeaIteration(imageBig, imageUnchanged, 8);
//	performNewIdeaIteration(imageBuffer, imageBig, 8);
//	performNewIdeaIteration(imageBig, imageBuffer, 8);
//	performNewIdeaIteration(imageBuffer, imageBig, 8);
//	performNewIdeaIteration(imageBig, imageBuffer, 8);
//
//    cuda_call(cudaMemcpy, device_image_a, imageSmall->data,
//              pixels_in_image * bytes_per_pixel, cudaMemcpyHostToDevice);
//
//    cuda_call(cudaMemcpy, device_image_b, imageBig->data,
//              pixels_in_image * bytes_per_pixel, cudaMemcpyHostToDevice);
//
//    cuda_launch(performNewIdeaFinalizationGPU, 1200, 240,
//                1920, device_image_a, device_image_b, device_image_output);
//
//    cuda_call(cudaMemcpy,
//              imageOut->data,
//              device_image_output,
//              pixels_in_image,
//              cudaMemcpyDeviceToHost);
//
//	if(argc > 1) {
//		writePPM("flower_medium.ppm", imageOut);
//	} else {
//		writeStreamPPM(stdout, imageOut);
//	}

    cuda_call(cudaFree, device_image_a);
    cuda_call(cudaFree, device_image_b);
    cuda_call(cudaFree, device_image_output);

	// free all memory structures
	freeImage(imageUnchanged);
	freeImage(imageBuffer);
	freeImage(imageSmall);
	freeImage(imageBig);
	free(imageOut->data);
	free(imageOut);
	free(image->data);
	free(image);

	return 0;
}

