#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

#include <cmath>
#include <iostream>
#include <algorithm>
#include <vector>

#include "ppm.h"

// Image from:
// http://7-themes.com/6971875-funny-flowers-pictures.html

struct  AccuratePixel  {
    float red,green,blue;
};

struct AccurateImage  {
     int x, y;
     AccuratePixel *data;
};

// Convert ppm to high precision format.
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

template <int size, float AccuratePixel::*color>
inline void doHorizontalStart(AccurateImage *out, AccurateImage *in, int y)
{
	for (int x = 0; x != size; ++x)
	{
		decltype(AccuratePixel::red) sum{};
		for (int s = 0; s != x + size + 1; ++s)
		{
			sum += in->data[y * in->x + s].*color;
		}
		int div = (size + x + 1);
		float scaler = (float)(2 * size + 1) / div;
		out->data[y * in->x + x].*color = sum * scaler;
	}
}

template <int size, float AccuratePixel::*color>
inline void doHorizontalMiddle(AccurateImage *out, AccurateImage *in, int y)
{
	for (int x = 0; x != in->x - 2 * size; ++x)
	{
		decltype(AccuratePixel::red) sum{};
		for (int s = 0; s != 2 * size + 1; ++s)
		{
			sum += in->data[y * in->x + x + s].*color;
		}
		out->data[y * in->x + x + size].*color = sum;
	}
}

template <int size, float AccuratePixel::*color>
inline void doHorizontalEnd(AccurateImage *out, AccurateImage *in, int y)
{
	for (int x = in->x - 2 * size; x != in->x - size; ++x)
	{
		decltype(AccuratePixel::red) sum{};
		for (int s = 0; s != in->x - x; ++s)
		{
			sum += in->data[y * in->x + x + s].*color;
		}
		int div = (in->x - x);
		float scaler = (float)(2 * size + 1) / div;
		out->data[y * in->x + x + size].*color = sum * scaler;
	}
}

template <int size, float AccuratePixel::*color>
inline void doVerticalStart(AccurateImage *out, AccurateImage *in, int x)
{
	for (int y = 0; y != size; ++y)
	{
		decltype(AccuratePixel::red) sum{};
		for (int s = 0; s != y + size + 1; ++s)
		{
			sum += in->data[s * in->x + x].*color;
		}
		int div = (size + y + 1);
		float scaler = (float)(2 * size + 1) / div;
		out->data[y * in->x + x].*color = sum * scaler;
	}
}

template <int size, float AccuratePixel::*color>
inline void doVerticalMiddle(AccurateImage *out, AccurateImage *in, int x)
{
	for (int y = 0; y != in->y - 2 * size; ++y)
	{
		decltype(AccuratePixel::red) sum{};
		for (int s = 0; s != 2 * size + 1; ++s)
		{
			sum += in->data[(y + s) * in->x + x].*color;
		}
		out->data[(y + size) * in->x + x].*color = sum;
	}
}

template <int size, float AccuratePixel::*color>
inline void doVerticalEnd(AccurateImage *out, AccurateImage *in, int x)
{
	for (int y = in->y - 2 * size; y != in->y - size; ++y)
	{
		decltype(AccuratePixel::red) sum{};
		for (int s = 0; s != in->y - y; ++s)
		{
			sum += in->data[(y + s) * in->x + x].*color;
		}
		int div = (in->y - y);
		float scaler = (float)(2 * size + 1) / div;
		out->data[(y + size) * in->x + x].*color = sum * scaler;
	}
}

template <int size, int iterations, float AccuratePixel::*color>
inline void DECIMATE(AccurateImage *out)
{
	for (int y = 0; y != out->y; ++y)
	{
		for (int x = 0; x != out->x; ++x)
		{
			//float div = std::pow(std::min(size, x) + std::min(size, out->x - x - 1) + 1, iterations) * std::pow(std::min(size, y) + std::min(size, out->y - y - 1) + 1, iterations);
			float div = std::pow(2 * size + 1, 2 * iterations);
			out->data[y * out->x + x].*color /= div;
		}
	}
}

// Perform the final step, and return it as ppm.
PPMImage * performNewIdeaFinalization(AccurateImage *imageInSmall, AccurateImage *imageInLarge) {
	PPMImage *imageOut;
	imageOut = (PPMImage *)malloc(sizeof(PPMImage));
	imageOut->data = (PPMPixel*)malloc(imageInSmall->x * imageInSmall->y * sizeof(PPMPixel));
	
	imageOut->x = imageInSmall->x;
	imageOut->y = imageInSmall->y;
	
	for(int i = 0; i < imageInSmall->x * imageInSmall->y; i++) {
		int value = (imageInLarge->data[i].red - imageInSmall->data[i].red);
		if(value > 255)
			imageOut->data[i].red = 255;
		else if (value < -1.0) {
			value = 257.0+value;
			if(value > 255)
				imageOut->data[i].red = 255;
			else
				imageOut->data[i].red = floor(value);
		} else if (value > -1.0 && value < 0.0) {
			imageOut->data[i].red = 0;
		} else {
			imageOut->data[i].red = floor(value);
		}
		
		value = (imageInLarge->data[i].green - imageInSmall->data[i].green);
		if(value > 255)
			imageOut->data[i].green = 255;
		else if (value < -1.0) {
			value = 257.0+value;
			if(value > 255)
				imageOut->data[i].green = 255;
			else
				imageOut->data[i].green = floor(value);
		} else if (value > -1.0 && value < 0.0) {
			imageOut->data[i].green = 0;
		} else {
			imageOut->data[i].green = floor(value);
		}
		
		value = (imageInLarge->data[i].blue - imageInSmall->data[i].blue);
		if(value > 255)
			imageOut->data[i].blue = 255;
		else if (value < -1.0) {
			value = 257.0+value;
			if(value > 255)
				imageOut->data[i].blue = 255;
			else
				imageOut->data[i].blue = floor(value);
		} else if (value > -1.0 && value < 0.0) {
			imageOut->data[i].blue = 0;
		} else {
			imageOut->data[i].blue = floor(value);
		}
	}
	
	
	return imageOut;
}

template <int size, float AccuratePixel::*color>
void performNewIdeaIteration(AccurateImage *first, AccurateImage *second)
{
	for (int y = 0; y != first->y; ++y)
	{
		doHorizontalStart<size, color>(first, second, y);
		doHorizontalMiddle<size, color>(first, second, y);
		doHorizontalEnd<size, color>(first, second, y);
	}

	for (int x = 0; x != first->x; ++x)
	{
		doVerticalStart<size, color>(second, first, x);
		doVerticalMiddle<size, color>(second, first, x);
		doVerticalEnd<size, color>(second, first, x);
	}
}

int main(int argc, char** argv) {
	PPMImage *image;
	
	if(argc > 1) {
		image = readPPM("flower.ppm");
	} else {
		image = readStreamPPM(stdin);
	}

	AccurateImage *imageAccurate1_tiny = convertImageToNewFormat(image);
	AccurateImage *imageAccurate2_tiny = convertImageToNewFormat(image);

	// Process the tiny case:
	performNewIdeaIteration<2, &AccuratePixel::red>(imageAccurate2_tiny, imageAccurate1_tiny);
	performNewIdeaIteration<2, &AccuratePixel::red>(imageAccurate2_tiny, imageAccurate1_tiny);
	performNewIdeaIteration<2, &AccuratePixel::red>(imageAccurate2_tiny, imageAccurate1_tiny);
	performNewIdeaIteration<2, &AccuratePixel::red>(imageAccurate2_tiny, imageAccurate1_tiny);
	performNewIdeaIteration<2, &AccuratePixel::red>(imageAccurate2_tiny, imageAccurate1_tiny);
	performNewIdeaIteration<2, &AccuratePixel::green>(imageAccurate2_tiny, imageAccurate1_tiny);
	performNewIdeaIteration<2, &AccuratePixel::green>(imageAccurate2_tiny, imageAccurate1_tiny);
	performNewIdeaIteration<2, &AccuratePixel::green>(imageAccurate2_tiny, imageAccurate1_tiny);
	performNewIdeaIteration<2, &AccuratePixel::green>(imageAccurate2_tiny, imageAccurate1_tiny);
	performNewIdeaIteration<2, &AccuratePixel::green>(imageAccurate2_tiny, imageAccurate1_tiny);
	performNewIdeaIteration<2, &AccuratePixel::blue>(imageAccurate2_tiny, imageAccurate1_tiny);
	performNewIdeaIteration<2, &AccuratePixel::blue>(imageAccurate2_tiny, imageAccurate1_tiny);
	performNewIdeaIteration<2, &AccuratePixel::blue>(imageAccurate2_tiny, imageAccurate1_tiny);
	performNewIdeaIteration<2, &AccuratePixel::blue>(imageAccurate2_tiny, imageAccurate1_tiny);
	performNewIdeaIteration<2, &AccuratePixel::blue>(imageAccurate2_tiny, imageAccurate1_tiny);
	DECIMATE<2, 5, &AccuratePixel::red>(imageAccurate1_tiny);
	DECIMATE<2, 5, &AccuratePixel::green>(imageAccurate1_tiny);
	DECIMATE<2, 5, &AccuratePixel::blue>(imageAccurate1_tiny);
	
	AccurateImage *imageAccurate1_small = convertImageToNewFormat(image);
	AccurateImage *imageAccurate2_small = convertImageToNewFormat(image);

	// Process the small case:
	performNewIdeaIteration<3, &AccuratePixel::red>(imageAccurate2_small, imageAccurate1_small);
	performNewIdeaIteration<3, &AccuratePixel::red>(imageAccurate2_small, imageAccurate1_small);
	performNewIdeaIteration<3, &AccuratePixel::red>(imageAccurate2_small, imageAccurate1_small);
	performNewIdeaIteration<3, &AccuratePixel::red>(imageAccurate2_small, imageAccurate1_small);
	performNewIdeaIteration<3, &AccuratePixel::red>(imageAccurate2_small, imageAccurate1_small);
	performNewIdeaIteration<3, &AccuratePixel::green>(imageAccurate2_small, imageAccurate1_small);
	performNewIdeaIteration<3, &AccuratePixel::green>(imageAccurate2_small, imageAccurate1_small);
	performNewIdeaIteration<3, &AccuratePixel::green>(imageAccurate2_small, imageAccurate1_small);
	performNewIdeaIteration<3, &AccuratePixel::green>(imageAccurate2_small, imageAccurate1_small);
	performNewIdeaIteration<3, &AccuratePixel::green>(imageAccurate2_small, imageAccurate1_small);
	performNewIdeaIteration<3, &AccuratePixel::blue>(imageAccurate2_small, imageAccurate1_small);
	performNewIdeaIteration<3, &AccuratePixel::blue>(imageAccurate2_small, imageAccurate1_small);
	performNewIdeaIteration<3, &AccuratePixel::blue>(imageAccurate2_small, imageAccurate1_small);
	performNewIdeaIteration<3, &AccuratePixel::blue>(imageAccurate2_small, imageAccurate1_small);
	performNewIdeaIteration<3, &AccuratePixel::blue>(imageAccurate2_small, imageAccurate1_small);
	DECIMATE<3, 5, &AccuratePixel::red>(imageAccurate1_small);
	DECIMATE<3, 5, &AccuratePixel::green>(imageAccurate1_small);
	DECIMATE<3, 5, &AccuratePixel::blue>(imageAccurate1_small);

	AccurateImage *imageAccurate1_medium = convertImageToNewFormat(image);
	AccurateImage *imageAccurate2_medium = convertImageToNewFormat(image);

	// Process the medium case:
	performNewIdeaIteration<5, &AccuratePixel::red>(imageAccurate2_medium, imageAccurate1_medium);
	performNewIdeaIteration<5, &AccuratePixel::red>(imageAccurate2_medium, imageAccurate1_medium);
	performNewIdeaIteration<5, &AccuratePixel::red>(imageAccurate2_medium, imageAccurate1_medium);
	performNewIdeaIteration<5, &AccuratePixel::red>(imageAccurate2_medium, imageAccurate1_medium);
	performNewIdeaIteration<5, &AccuratePixel::red>(imageAccurate2_medium, imageAccurate1_medium);
	performNewIdeaIteration<5, &AccuratePixel::green>(imageAccurate2_medium, imageAccurate1_medium);
	performNewIdeaIteration<5, &AccuratePixel::green>(imageAccurate2_medium, imageAccurate1_medium);
	performNewIdeaIteration<5, &AccuratePixel::green>(imageAccurate2_medium, imageAccurate1_medium);
	performNewIdeaIteration<5, &AccuratePixel::green>(imageAccurate2_medium, imageAccurate1_medium);
	performNewIdeaIteration<5, &AccuratePixel::green>(imageAccurate2_medium, imageAccurate1_medium);
	performNewIdeaIteration<5, &AccuratePixel::blue>(imageAccurate2_medium, imageAccurate1_medium);
	performNewIdeaIteration<5, &AccuratePixel::blue>(imageAccurate2_medium, imageAccurate1_medium);
	performNewIdeaIteration<5, &AccuratePixel::blue>(imageAccurate2_medium, imageAccurate1_medium);
	performNewIdeaIteration<5, &AccuratePixel::blue>(imageAccurate2_medium, imageAccurate1_medium);
	performNewIdeaIteration<5, &AccuratePixel::blue>(imageAccurate2_medium, imageAccurate1_medium);
	DECIMATE<5, 5, &AccuratePixel::red>(imageAccurate1_medium);
	DECIMATE<5, 5, &AccuratePixel::green>(imageAccurate1_medium);
	DECIMATE<5, 5, &AccuratePixel::blue>(imageAccurate1_medium);

	AccurateImage *imageAccurate1_large = convertImageToNewFormat(image);
	AccurateImage *imageAccurate2_large = convertImageToNewFormat(image);

	// Process the large case:
	performNewIdeaIteration<8, &AccuratePixel::red>(imageAccurate2_large, imageAccurate1_large);
	performNewIdeaIteration<8, &AccuratePixel::red>(imageAccurate2_large, imageAccurate1_large);
	performNewIdeaIteration<8, &AccuratePixel::red>(imageAccurate2_large, imageAccurate1_large);
	performNewIdeaIteration<8, &AccuratePixel::red>(imageAccurate2_large, imageAccurate1_large);
	performNewIdeaIteration<8, &AccuratePixel::red>(imageAccurate2_large, imageAccurate1_large);
	performNewIdeaIteration<8, &AccuratePixel::green>(imageAccurate2_large, imageAccurate1_large);
	performNewIdeaIteration<8, &AccuratePixel::green>(imageAccurate2_large, imageAccurate1_large);
	performNewIdeaIteration<8, &AccuratePixel::green>(imageAccurate2_large, imageAccurate1_large);
	performNewIdeaIteration<8, &AccuratePixel::green>(imageAccurate2_large, imageAccurate1_large);
	performNewIdeaIteration<8, &AccuratePixel::green>(imageAccurate2_large, imageAccurate1_large);
	performNewIdeaIteration<8, &AccuratePixel::blue>(imageAccurate2_large, imageAccurate1_large);
	performNewIdeaIteration<8, &AccuratePixel::blue>(imageAccurate2_large, imageAccurate1_large);
	performNewIdeaIteration<8, &AccuratePixel::blue>(imageAccurate2_large, imageAccurate1_large);
	performNewIdeaIteration<8, &AccuratePixel::blue>(imageAccurate2_large, imageAccurate1_large);
	performNewIdeaIteration<8, &AccuratePixel::blue>(imageAccurate2_large, imageAccurate1_large);
	DECIMATE<8, 5, &AccuratePixel::red>(imageAccurate1_large);
	DECIMATE<8, 5, &AccuratePixel::green>(imageAccurate1_large);
	DECIMATE<8, 5, &AccuratePixel::blue>(imageAccurate1_large);

	//// Save the images.
	PPMImage *final_tiny = performNewIdeaFinalization(imageAccurate1_tiny, imageAccurate1_small);
	PPMImage *final_small = performNewIdeaFinalization(imageAccurate1_small, imageAccurate1_medium);
	PPMImage *final_medium = performNewIdeaFinalization(imageAccurate1_medium, imageAccurate1_large);

	if (argc > 1) {
		writePPM("flower_tiny.ppm", final_tiny);
		writePPM("flower_small.ppm", final_small);
		writePPM("flower_medium.ppm", final_medium);
	} else {
		writeStreamPPM(stdout, final_tiny);
		writeStreamPPM(stdout, final_small);
		writeStreamPPM(stdout, final_medium);
	}

	return 0;
}
