#include <stdexcept>
#include <string>
#include <iostream>

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
        throw std::runtime_error(std::string(function) + " failed: " + cudaGetErrorString(error));
    }
}

// Image from:
// http://7-themes.com/6971875-funny-flowers-pictures.html

// TODO: You must implement this
// The handout code is much simpler than the MPI/OpenMP versions
//__global__ void performNewIdeaIterationGPU( ... ) { ... }

// TODO: You should implement this
//__global__ void performNewIdeaFinalizationGPU( ... ) { ... }

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

// Image must have width divisible by 8 pixels
__global__
void
performNewIdeaFinalizationGPU(
    const int            image_width,
    float* const         image_input_small,
    float* const         image_input_large,
    std::uint64_t* const image_output)
{
    const int offset = blockIdx.x * image_width * 24 + 24 * threadIdx.x;

    // Input:  24 * float values = 24 * 4 bytes = 96 bytes (8 bytes * 12)
    // Output: 24 bytes (8 bytes * 3)
    for (int i = 0; i < 24; ++i)
    {
        image_output[offset + i] = computeFinalPixelValue(
            image_input_large[offset + i] - image_input_small[offset + i]);
    }

    const int output_offset = blockIdx.x * image_width + threadIdx.x;

    image_output[offset + 0] = (computeFinalPixelValue(image_input_large[offset + 0] - image_input_small[offset + 0]) << 56) |
                               (computeFinalPixelValue(image_input_large[offset + 1] - image_input_small[offset + 1]) << 48) |
                               (computeFinalPixelValue(image_input_large[offset + 2] - image_input_small[offset + 2]) << 40) |
                               (computeFinalPixelValue(image_input_large[offset + 3] - image_input_small[offset + 3]) << 32) |
                               (computeFinalPixelValue(image_input_large[offset + 4] - image_input_small[offset + 4]) << 24) |
                               (computeFinalPixelValue(image_input_large[offset + 5] - image_input_small[offset + 5]) << 16) |
                               (computeFinalPixelValue(image_input_large[offset + 6] - image_input_small[offset + 6]) <<  8) |
                               (computeFinalPixelValue(image_input_large[offset + 7] - image_input_small[offset + 7]) <<  0);

}

// TODO: You should implement this
//__global__ void convertImageToNewFormatGPU( ... ) { ... }

// Perhaps some extra kernels will be practical as well?
//__global__ void ...GPU( ... ) { ... }

typedef struct {
     float red,green,blue;
} AccuratePixel;

typedef struct {
     int x, y;
     AccuratePixel *data;
} AccurateImage;

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
    cuda_call(cudaMalloc, &device_image_output, pixels_in_image);

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

	// Process the tiny case:
	performNewIdeaIteration(imageSmall, imageUnchanged, 2);
	performNewIdeaIteration(imageBuffer, imageSmall, 2);
	performNewIdeaIteration(imageSmall, imageBuffer, 2);
	performNewIdeaIteration(imageBuffer, imageSmall, 2);
	performNewIdeaIteration(imageSmall, imageBuffer, 2);

	// Process the small case:
	performNewIdeaIteration(imageBig, imageUnchanged,3);
	performNewIdeaIteration(imageBuffer, imageBig,3);
	performNewIdeaIteration(imageBig, imageBuffer,3);
	performNewIdeaIteration(imageBuffer, imageBig,3);
	performNewIdeaIteration(imageBig, imageBuffer,3);
    
    std::cout << "N " << device_image_a << std::endl;

    cuda_call(cudaMemcpy, device_image_a, imageSmall->data,
              pixels_in_image * bytes_per_pixel, cudaMemcpyHostToDevice);
    
    std::cout << "X" << std::endl;

    cuda_call(cudaMemcpy, device_image_b, imageBig->data,
              pixels_in_image * bytes_per_pixel, cudaMemcpyHostToDevice);
    
    std::cout << "Y" << std::endl;

    cuda_launch(performNewIdeaFinalizationGPU, 200, 240,
                1920, device_image_a, device_image_b, device_image_output);
    
    std::cout << "Z" << std::endl;

    cuda_call(cudaMemcpy,
              imageOut->data,
              device_image_output,
              pixels_in_image,
              cudaMemcpyDeviceToHost);
    
    std::cout << "1" << std::endl;

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

