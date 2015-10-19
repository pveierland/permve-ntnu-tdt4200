#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

#include "ppm.h"

#include <iostream>
#include <vector>

// Image from:
// http://7-themes.com/6971875-funny-flowers-pictures.html

using ValueType = float;

struct AccuratePixel  {
	ValueType red, green, blue;
};

struct AccurateImage {
	int x, y;
	AccuratePixel *data;
};

void transpose(AccurateImage& dest, AccurateImage& source)
{
	for (int y = 0; y != source.y; ++y)
	{
		for (int x = 0; x != source.x; ++x)
		{
			dest.data[x * source.y + y] = source.data[y * source.x + x];
		}
	}
	dest.x = source.y;
	dest.y = source.x;
}

// Convert ppm to high precision format.
AccurateImage *convertImageToNewFormat(PPMImage *image) {
	// Make a copy
	AccurateImage *imageAccurate;
	imageAccurate = (AccurateImage *)malloc(sizeof(AccurateImage));
	imageAccurate->data = (AccuratePixel*)malloc(image->x * image->y * sizeof(AccuratePixel));
	for (int i = 0; i < image->x * image->y; i++) {
		imageAccurate->data[i].red = (double)image->data[i].red;
		imageAccurate->data[i].green = (double)image->data[i].green;
		imageAccurate->data[i].blue = (double)image->data[i].blue;
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
			int index = y * in->x + s;
			sum += in->data[index].*color;
		}
		int div = x + size + 1;
		out->data[y * in->x + x].*color = sum / div;
	}
}

template <int size, float AccuratePixel::*color>
inline void doHorizontalMiddle(AccurateImage *out, AccurateImage *in, int y, int fragment_x, int fragment_width)
{
	for (int x = fragment_x; x != fragment_x + fragment_width; ++x)
	{
		decltype(AccuratePixel::red) sum{};
		for (int s = 0; s != 2 * size + 1; ++s)
		{
			int index = y * in->x + x + s;
			sum += in->data[index].*color;
		}
		int div = (2 * size + 1);
		out->data[y * in->x + x + size].*color = sum / div;
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
			int index = y * in->x + x + s;
			sum += in->data[index].*color;
		}
		int div = (in->x - x);
		out->data[y * in->x + x + size].*color = sum / div;
	}
}

// Perform the new idea:
//template <float AccuratePixel::*color>
//void performNewIdeaIteration(AccurateImage *out, AccurateImage *in, int width, int height, int size) {
//
//	// Iterate over each pixel
//	for (int senterY = 0; senterY < in->y; senterY++) {
//		for (int senterX = 0; senterX < in->x; senterX++) {
//
//			// For each pixel we compute the magic number
//			float sum = 0;
//			float countIncluded = 0;
//
//			for (int y = -size; y <= size; y++) {
//				
//				for (int x = -size; x <= size; x++)
//				{
//					int currentX = senterX;// +x;
//					int currentY = senterY;// +y;
//
//					// Check if we are outside the bounds
//					if (currentX < 0)
//						continue;
//					if (currentX >= in->x)
//						continue;
//					if (currentY < 0)
//						continue;
//					if (currentY >= in->y)
//						continue;
//
//					// Now we can begin
//					int numberOfValuesInEachRow = in->x;
//					int offsetOfThePixel = (numberOfValuesInEachRow * currentY + currentX);
//					sum += in->data[offsetOfThePixel].*color;
//
//					// Keep track of how many values we have included
//					countIncluded += 1;
//				}
//			}
//
//			// Now we compute the final value
//			double value = sum / countIncluded;
//
//			// Update the output image
//			int numberOfValuesInEachRow = out->x; // R, G and B
//			int offsetOfThePixel = (numberOfValuesInEachRow * senterY + senterX);
//
//			out->data[offsetOfThePixel].*color = value;
//		}
//	}
//}

void clever(AccurateImage* out, AccurateImage* in, int offset_y, int offset_x, int width, int height)
{
	for (int y = offset_y; y != offset_y + height; ++y)
	{
		out->data[(offset_y + y) * width + (offset_x)].red =
			   (in->data[(offset_x + y) * width + (offset_x)].red +
				in->data[(offset_x + y) * width + (offset_x + 1)].red +
				in->data[(offset_x + y) * width + (offset_x + 2)].red +
				in->data[(offset_x + y) * width + (offset_x + 3)].red +
				in->data[(offset_x + y) * width + (offset_x + 3)].red +
				in->data[(offset_x + y) * width + (offset_x + 4)].red +
				in->data[(offset_x + y) * width + (offset_x + 5)].red +
				in->data[(offset_x + y) * width + (offset_x + 6)].red +
				in->data[(offset_x + y) * width + (offset_x + 7)].red) / 9.0;

		out->data[(offset_y + y) * width + (offset_x)].blue =
			(in->data[(offset_x + y) * width + (offset_x)].red +
				in->data[(offset_x + y) * width + (offset_x + 1)].red +
				in->data[(offset_x + y) * width + (offset_x + 2)].red +
				in->data[(offset_x + y) * width + (offset_x + 3)].red +
				in->data[(offset_x + y) * width + (offset_x + 3)].red +
				in->data[(offset_x + y) * width + (offset_x + 4)].red +
				in->data[(offset_x + y) * width + (offset_x + 5)].red +
				in->data[(offset_x + y) * width + (offset_x + 6)].red +
				in->data[(offset_x + y) * width + (offset_x + 7)].red) / 9.0;
	}

}

void performNewIdeaIteration(AccurateImage *imageOut, AccurateImage *imageIn, int colourType, int size) {

	// Iterate over each pixel
	
	

	for (int senterX = 0; senterX < imageIn->x; senterX++) {

		for (int senterY = 0; senterY < imageIn->y; senterY++) {

			// For each pixel we compute the magic number
			double sum = 0;
			int countIncluded = 0;

			
			
			for (int x = -size; x <= size; x++) {

				for (int y = -size; y <= size; y++) {

					int currentX = senterX + x;
					int currentY = senterY + y;

					// Check if we are outside the bounds
					if (currentX < 0)
						continue;
					if (currentX >= imageIn->x)
						continue;
					if (currentY < 0)
						continue;
					if (currentY >= imageIn->y)
						continue;

					// Now we can begin
					int numberOfValuesInEachRow = imageIn->x;
					int offsetOfThePixel = (numberOfValuesInEachRow * currentY + currentX);
					if (colourType == 0)
						sum += imageIn->data[offsetOfThePixel].red;
					if (colourType == 1)
						sum += imageIn->data[offsetOfThePixel].green;
					if (colourType == 2)
						sum += imageIn->data[offsetOfThePixel].blue;

					// Keep track of how many values we have included
					countIncluded++;
				}

			}

			// Now we compute the final value
			double value = sum / countIncluded;


			// Update the output image
			int numberOfValuesInEachRow = imageOut->x; // R, G and B
			int offsetOfThePixel = (numberOfValuesInEachRow * senterY + senterX);
			if (colourType == 0)
				imageOut->data[offsetOfThePixel].red = value;
			if (colourType == 1)
				imageOut->data[offsetOfThePixel].green = value;
			if (colourType == 2)
				imageOut->data[offsetOfThePixel].blue = value;
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

	for (int i = 0; i < imageInSmall->x * imageInSmall->y; i++) {
		double value = (imageInLarge->data[i].red - imageInSmall->data[i].red);
		if (value > 255)
			imageOut->data[i].red = 255;
		else if (value < -1.0) {
			value = 257.0 + value;
			if (value > 255)
				imageOut->data[i].red = 255;
			else
				imageOut->data[i].red = floor(value);
		}
		else if (value > -1.0 && value < 0.0) {
			imageOut->data[i].red = 0;
		}
		else {
			imageOut->data[i].red = floor(value);
		}

		value = (imageInLarge->data[i].green - imageInSmall->data[i].green);
		if (value > 255)
			imageOut->data[i].green = 255;
		else if (value < -1.0) {
			value = 257.0 + value;
			if (value > 255)
				imageOut->data[i].green = 255;
			else
				imageOut->data[i].green = floor(value);
		}
		else if (value > -1.0 && value < 0.0) {
			imageOut->data[i].green = 0;
		}
		else {
			imageOut->data[i].green = floor(value);
		}

		value = (imageInLarge->data[i].blue - imageInSmall->data[i].blue);
		if (value > 255)
			imageOut->data[i].blue = 255;
		else if (value < -1.0) {
			value = 257.0 + value;
			if (value > 255)
				imageOut->data[i].blue = 255;
			else
				imageOut->data[i].blue = floor(value);
		}
		else if (value > -1.0 && value < 0.0) {
			imageOut->data[i].blue = 0;
		}
		else {
			imageOut->data[i].blue = floor(value);
		}
	}


	return imageOut;
}


int main(int argc, char** argv) {

	PPMImage *image;

	if (argc > 1) {
		image = readPPM("flower.ppm");
	}
	else {
		image = readStreamPPM(stdin);
	}

	AccurateImage *imageAccurate1_tiny = convertImageToNewFormat(image);
	AccurateImage *imageAccurate2_tiny = convertImageToNewFormat(image);

	for (int iterations = 0; iterations != 5; ++iterations)
	{
		for (int f_y = 0; f_y != 10; ++f_y)
		{
			for (int y = 0; y != 120; ++y)
			{
				doHorizontalStart<2, &AccuratePixel::red>(imageAccurate2_tiny, imageAccurate1_tiny, f_y * 120 + y);
				doHorizontalMiddle<2, &AccuratePixel::red>(imageAccurate2_tiny, imageAccurate1_tiny, f_y * 120 + y, 0, 396);
				doHorizontalStart<2, &AccuratePixel::green>(imageAccurate2_tiny, imageAccurate1_tiny, f_y * 120 + y);
				doHorizontalMiddle<2, &AccuratePixel::green>(imageAccurate2_tiny, imageAccurate1_tiny, f_y * 120 + y, 0, 396);
				doHorizontalStart<2, &AccuratePixel::blue>(imageAccurate2_tiny, imageAccurate1_tiny, f_y * 120 + y);
				doHorizontalMiddle<2, &AccuratePixel::blue>(imageAccurate2_tiny, imageAccurate1_tiny, f_y * 120 + y, 0, 396);
			}

			for (int y = 0; y != 120; ++y)
			{
				doHorizontalMiddle<2, &AccuratePixel::red>(imageAccurate2_tiny, imageAccurate1_tiny, f_y * 120 + y, 396, 396);
				doHorizontalMiddle<2, &AccuratePixel::green>(imageAccurate2_tiny, imageAccurate1_tiny, f_y * 120 + y, 396, 396);
				doHorizontalMiddle<2, &AccuratePixel::blue>(imageAccurate2_tiny, imageAccurate1_tiny, f_y * 120 + y, 396, 396);
			}

			for (int y = 0; y != 120; ++y)
			{
				doHorizontalMiddle<2, &AccuratePixel::red>(imageAccurate2_tiny, imageAccurate1_tiny, f_y * 120 + y, 792, 396);
				doHorizontalMiddle<2, &AccuratePixel::green>(imageAccurate2_tiny, imageAccurate1_tiny, f_y * 120 + y, 792, 396);
				doHorizontalMiddle<2, &AccuratePixel::blue>(imageAccurate2_tiny, imageAccurate1_tiny, f_y * 120 + y, 792, 396);
			}

			for (int y = 0; y != 120; ++y)
			{
				doHorizontalMiddle<2, &AccuratePixel::red>(imageAccurate2_tiny, imageAccurate1_tiny, f_y * 120 + y, 1188, 396);
				doHorizontalMiddle<2, &AccuratePixel::green>(imageAccurate2_tiny, imageAccurate1_tiny, f_y * 120 + y, 1188, 396);
				doHorizontalMiddle<2, &AccuratePixel::blue>(imageAccurate2_tiny, imageAccurate1_tiny, f_y * 120 + y, 1188, 396);
			}

			for (int y = 0; y != 120; ++y)
			{
				doHorizontalMiddle<2, &AccuratePixel::red>(imageAccurate2_tiny, imageAccurate1_tiny, f_y * 120 + y, 1584, 392);
				doHorizontalEnd<2, &AccuratePixel::red>(imageAccurate2_tiny, imageAccurate1_tiny, f_y * 120 + y);
				doHorizontalMiddle<2, &AccuratePixel::green>(imageAccurate2_tiny, imageAccurate1_tiny, f_y * 120 + y, 1584, 392);
				doHorizontalEnd<2, &AccuratePixel::green>(imageAccurate2_tiny, imageAccurate1_tiny, f_y * 120 + y);
				doHorizontalMiddle<2, &AccuratePixel::blue>(imageAccurate2_tiny, imageAccurate1_tiny, f_y * 120 + y, 1584, 392);
				doHorizontalEnd<2, &AccuratePixel::blue>(imageAccurate2_tiny, imageAccurate1_tiny, f_y * 120 + y);
			}
		}

		transpose(*imageAccurate1_tiny, *imageAccurate2_tiny);

		for (int f_y = 0; f_y != 10; ++f_y)
		{
			for (int y = 0; y != 198; ++y)
			{
				doHorizontalStart<2, &AccuratePixel::red>(imageAccurate2_tiny, imageAccurate1_tiny, f_y * 198 + y);
				doHorizontalMiddle<2, &AccuratePixel::red>(imageAccurate2_tiny, imageAccurate1_tiny, f_y * 198 + y, 0, 240);
				doHorizontalStart<2, &AccuratePixel::green>(imageAccurate2_tiny, imageAccurate1_tiny, f_y * 198 + y);
				doHorizontalMiddle<2, &AccuratePixel::green>(imageAccurate2_tiny, imageAccurate1_tiny, f_y * 198 + y, 0, 240);
				doHorizontalStart<2, &AccuratePixel::blue>(imageAccurate2_tiny, imageAccurate1_tiny, f_y * 198 + y);
				doHorizontalMiddle<2, &AccuratePixel::blue>(imageAccurate2_tiny, imageAccurate1_tiny, f_y * 198 + y, 0, 240);
			}

			for (int y = 0; y != 198; ++y)
			{
				doHorizontalMiddle<2, &AccuratePixel::red>(imageAccurate2_tiny, imageAccurate1_tiny, f_y * 198 + y, 240, 240);
				doHorizontalMiddle<2, &AccuratePixel::green>(imageAccurate2_tiny, imageAccurate1_tiny, f_y * 198 + y, 240, 240);
				doHorizontalMiddle<2, &AccuratePixel::blue>(imageAccurate2_tiny, imageAccurate1_tiny, f_y * 198 + y, 240, 240);
			}

			for (int y = 0; y != 198; ++y)
			{
				doHorizontalMiddle<2, &AccuratePixel::red>(imageAccurate2_tiny, imageAccurate1_tiny, f_y * 198 + y, 480, 240);
				doHorizontalMiddle<2, &AccuratePixel::green>(imageAccurate2_tiny, imageAccurate1_tiny, f_y * 198 + y, 480, 240);
				doHorizontalMiddle<2, &AccuratePixel::blue>(imageAccurate2_tiny, imageAccurate1_tiny, f_y * 198 + y, 480, 240);
			}

			for (int y = 0; y != 198; ++y)
			{
				doHorizontalMiddle<2, &AccuratePixel::red>(imageAccurate2_tiny, imageAccurate1_tiny, f_y * 198 + y, 720, 240);
				doHorizontalMiddle<2, &AccuratePixel::green>(imageAccurate2_tiny, imageAccurate1_tiny, f_y * 198 + y, 720, 240);
				doHorizontalMiddle<2, &AccuratePixel::blue>(imageAccurate2_tiny, imageAccurate1_tiny, f_y * 198 + y, 720, 240);
			}

			for (int y = 0; y != 198; ++y)
			{
				doHorizontalMiddle<2, &AccuratePixel::red>(imageAccurate2_tiny, imageAccurate1_tiny, f_y * 198 + y, 960, 240);
				doHorizontalMiddle<2, &AccuratePixel::green>(imageAccurate2_tiny, imageAccurate1_tiny, f_y * 198 + y, 960, 240);
				doHorizontalMiddle<2, &AccuratePixel::blue>(imageAccurate2_tiny, imageAccurate1_tiny, f_y * 198 + y, 960, 240);
			}

			for (int y = 0; y != 198; ++y)
			{
				doHorizontalMiddle<2, &AccuratePixel::red>(imageAccurate2_tiny, imageAccurate1_tiny, f_y * 198 + y, 960, 236);
				doHorizontalEnd<2, &AccuratePixel::red>(imageAccurate2_tiny, imageAccurate1_tiny, f_y * 198 + y);
				doHorizontalMiddle<2, &AccuratePixel::green>(imageAccurate2_tiny, imageAccurate1_tiny, f_y * 198 + y, 960, 236);
				doHorizontalEnd<2, &AccuratePixel::green>(imageAccurate2_tiny, imageAccurate1_tiny, f_y * 198 + y);
				doHorizontalMiddle<2, &AccuratePixel::blue>(imageAccurate2_tiny, imageAccurate1_tiny, f_y * 198 + y, 960, 236);
				doHorizontalEnd<2, &AccuratePixel::blue>(imageAccurate2_tiny, imageAccurate1_tiny, f_y * 198 + y);
			}
		}

		transpose(*imageAccurate1_tiny, *imageAccurate2_tiny);
	}

	AccurateImage *imageAccurate1_small = convertImageToNewFormat(image);
	AccurateImage *imageAccurate2_small = convertImageToNewFormat(image);

	for (int iterations = 0; iterations != 5; ++iterations)
	{
		for (int f_y = 0; f_y != 10; ++f_y)
		{
			for (int y = 0; y != 120; ++y)
			{
				doHorizontalStart<3, &AccuratePixel::red>(imageAccurate2_small, imageAccurate1_small, f_y * 120 + y);
				doHorizontalMiddle<3, &AccuratePixel::red>(imageAccurate2_small, imageAccurate1_small, f_y * 120 + y, 0, 396);
				doHorizontalStart<3, &AccuratePixel::green>(imageAccurate2_small, imageAccurate1_small, f_y * 120 + y);
				doHorizontalMiddle<3, &AccuratePixel::green>(imageAccurate2_small, imageAccurate1_small, f_y * 120 + y, 0, 396);
				doHorizontalStart<3, &AccuratePixel::blue>(imageAccurate2_small, imageAccurate1_small, f_y * 120 + y);
				doHorizontalMiddle<3, &AccuratePixel::blue>(imageAccurate2_small, imageAccurate1_small, f_y * 120 + y, 0, 396);
			}

			for (int y = 0; y != 120; ++y)
			{
				doHorizontalMiddle<3, &AccuratePixel::red>(imageAccurate2_small, imageAccurate1_small, f_y * 120 + y, 396, 396);
				doHorizontalMiddle<3, &AccuratePixel::green>(imageAccurate2_small, imageAccurate1_small, f_y * 120 + y, 396, 396);
				doHorizontalMiddle<3, &AccuratePixel::blue>(imageAccurate2_small, imageAccurate1_small, f_y * 120 + y, 396, 396);
			}

			for (int y = 0; y != 120; ++y)
			{
				doHorizontalMiddle<3, &AccuratePixel::red>(imageAccurate2_small, imageAccurate1_small, f_y * 120 + y, 792, 396);
				doHorizontalMiddle<3, &AccuratePixel::green>(imageAccurate2_small, imageAccurate1_small, f_y * 120 + y, 792, 396);
				doHorizontalMiddle<3, &AccuratePixel::blue>(imageAccurate2_small, imageAccurate1_small, f_y * 120 + y, 792, 396);
			}

			for (int y = 0; y != 120; ++y)
			{
				doHorizontalMiddle<3, &AccuratePixel::red>(imageAccurate2_small, imageAccurate1_small, f_y * 120 + y, 1188, 396);
				doHorizontalMiddle<3, &AccuratePixel::green>(imageAccurate2_small, imageAccurate1_small, f_y * 120 + y, 1188, 396);
				doHorizontalMiddle<3, &AccuratePixel::blue>(imageAccurate2_small, imageAccurate1_small, f_y * 120 + y, 1188, 396);
			}

			for (int y = 0; y != 120; ++y)
			{
				doHorizontalMiddle<3, &AccuratePixel::red>(imageAccurate2_small, imageAccurate1_small, f_y * 120 + y, 1584, 390);
				doHorizontalEnd<3, &AccuratePixel::red>(imageAccurate2_small, imageAccurate1_small, f_y * 120 + y);
				doHorizontalMiddle<3, &AccuratePixel::green>(imageAccurate2_small, imageAccurate1_small, f_y * 120 + y, 1584, 390);
				doHorizontalEnd<3, &AccuratePixel::green>(imageAccurate2_small, imageAccurate1_small, f_y * 120 + y);
				doHorizontalMiddle<3, &AccuratePixel::blue>(imageAccurate2_small, imageAccurate1_small, f_y * 120 + y, 1584, 390);
				doHorizontalEnd<3, &AccuratePixel::blue>(imageAccurate2_small, imageAccurate1_small, f_y * 120 + y);
			}
		}

		transpose(*imageAccurate1_small, *imageAccurate2_small);

		for (int f_y = 0; f_y != 10; ++f_y)
		{
			for (int y = 0; y != 198; ++y)
			{
				doHorizontalStart<3, &AccuratePixel::red>(imageAccurate2_small, imageAccurate1_small, f_y * 198 + y);
				doHorizontalMiddle<3, &AccuratePixel::red>(imageAccurate2_small, imageAccurate1_small, f_y * 198 + y, 0, 240);
				doHorizontalStart<3, &AccuratePixel::green>(imageAccurate2_small, imageAccurate1_small, f_y * 198 + y);
				doHorizontalMiddle<3, &AccuratePixel::green>(imageAccurate2_small, imageAccurate1_small, f_y * 198 + y, 0, 240);
				doHorizontalStart<3, &AccuratePixel::blue>(imageAccurate2_small, imageAccurate1_small, f_y * 198 + y);
				doHorizontalMiddle<3, &AccuratePixel::blue>(imageAccurate2_small, imageAccurate1_small, f_y * 198 + y, 0, 240);
			}

			for (int y = 0; y != 198; ++y)
			{
				doHorizontalMiddle<3, &AccuratePixel::red>(imageAccurate2_small, imageAccurate1_small, f_y * 198 + y, 240, 240);
				doHorizontalMiddle<3, &AccuratePixel::green>(imageAccurate2_small, imageAccurate1_small, f_y * 198 + y, 240, 240);
				doHorizontalMiddle<3, &AccuratePixel::blue>(imageAccurate2_small, imageAccurate1_small, f_y * 198 + y, 240, 240);
			}

			for (int y = 0; y != 198; ++y)
			{
				doHorizontalMiddle<3, &AccuratePixel::red>(imageAccurate2_small, imageAccurate1_small, f_y * 198 + y, 480, 240);
				doHorizontalMiddle<3, &AccuratePixel::green>(imageAccurate2_small, imageAccurate1_small, f_y * 198 + y, 480, 240);
				doHorizontalMiddle<3, &AccuratePixel::blue>(imageAccurate2_small, imageAccurate1_small, f_y * 198 + y, 480, 240);
			}

			for (int y = 0; y != 198; ++y)
			{
				doHorizontalMiddle<3, &AccuratePixel::red>(imageAccurate2_small, imageAccurate1_small, f_y * 198 + y, 720, 240);
				doHorizontalMiddle<3, &AccuratePixel::green>(imageAccurate2_small, imageAccurate1_small, f_y * 198 + y, 720, 240);
				doHorizontalMiddle<3, &AccuratePixel::blue>(imageAccurate2_small, imageAccurate1_small, f_y * 198 + y, 720, 240);
			}

			for (int y = 0; y != 198; ++y)
			{
				doHorizontalMiddle<3, &AccuratePixel::red>(imageAccurate2_small, imageAccurate1_small, f_y * 198 + y, 960, 240);
				doHorizontalMiddle<3, &AccuratePixel::green>(imageAccurate2_small, imageAccurate1_small, f_y * 198 + y, 960, 240);
				doHorizontalMiddle<3, &AccuratePixel::blue>(imageAccurate2_small, imageAccurate1_small, f_y * 198 + y, 960, 240);
			}

			for (int y = 0; y != 198; ++y)
			{
				doHorizontalMiddle<3, &AccuratePixel::red>(imageAccurate2_small, imageAccurate1_small, f_y * 198 + y, 960, 234);
				doHorizontalEnd<3, &AccuratePixel::red>(imageAccurate2_small, imageAccurate1_small, f_y * 198 + y);
				doHorizontalMiddle<3, &AccuratePixel::green>(imageAccurate2_small, imageAccurate1_small, f_y * 198 + y, 960, 234);
				doHorizontalEnd<3, &AccuratePixel::green>(imageAccurate2_small, imageAccurate1_small, f_y * 198 + y);
				doHorizontalMiddle<3, &AccuratePixel::blue>(imageAccurate2_small, imageAccurate1_small, f_y * 198 + y, 960, 234);
				doHorizontalEnd<3, &AccuratePixel::blue>(imageAccurate2_small, imageAccurate1_small, f_y * 198 + y);
			}
		}

		transpose(*imageAccurate1_small, *imageAccurate2_small);
	}

	AccurateImage *imageAccurate1_medium = convertImageToNewFormat(image);
	AccurateImage *imageAccurate2_medium = convertImageToNewFormat(image);

	for (int iterations = 0; iterations != 5; ++iterations)
	{
		for (int f_y = 0; f_y != 10; ++f_y)
		{
			for (int y = 0; y != 120; ++y)
			{
				doHorizontalStart<5, &AccuratePixel::red>(imageAccurate2_medium, imageAccurate1_medium, f_y * 120 + y);
				doHorizontalMiddle<5, &AccuratePixel::red>(imageAccurate2_medium, imageAccurate1_medium, f_y * 120 + y, 0, 396);
				doHorizontalStart<5, &AccuratePixel::green>(imageAccurate2_medium, imageAccurate1_medium, f_y * 120 + y);
				doHorizontalMiddle<5, &AccuratePixel::green>(imageAccurate2_medium, imageAccurate1_medium, f_y * 120 + y, 0, 396);
				doHorizontalStart<5, &AccuratePixel::blue>(imageAccurate2_medium, imageAccurate1_medium, f_y * 120 + y);
				doHorizontalMiddle<5, &AccuratePixel::blue>(imageAccurate2_medium, imageAccurate1_medium, f_y * 120 + y, 0, 396);
			}

			for (int y = 0; y != 120; ++y)
			{
				doHorizontalMiddle<5, &AccuratePixel::red>(imageAccurate2_medium, imageAccurate1_medium, f_y * 120 + y, 396, 396);
				doHorizontalMiddle<5, &AccuratePixel::green>(imageAccurate2_medium, imageAccurate1_medium, f_y * 120 + y, 396, 396);
				doHorizontalMiddle<5, &AccuratePixel::blue>(imageAccurate2_medium, imageAccurate1_medium, f_y * 120 + y, 396, 396);
			}

			for (int y = 0; y != 120; ++y)
			{
				doHorizontalMiddle<5, &AccuratePixel::red>(imageAccurate2_medium, imageAccurate1_medium, f_y * 120 + y, 792, 396);
				doHorizontalMiddle<5, &AccuratePixel::green>(imageAccurate2_medium, imageAccurate1_medium, f_y * 120 + y, 792, 396);
				doHorizontalMiddle<5, &AccuratePixel::blue>(imageAccurate2_medium, imageAccurate1_medium, f_y * 120 + y, 792, 396);
			}

			for (int y = 0; y != 120; ++y)
			{
				doHorizontalMiddle<5, &AccuratePixel::red>(imageAccurate2_medium, imageAccurate1_medium, f_y * 120 + y, 1188, 396);
				doHorizontalMiddle<5, &AccuratePixel::green>(imageAccurate2_medium, imageAccurate1_medium, f_y * 120 + y, 1188, 396);
				doHorizontalMiddle<5, &AccuratePixel::blue>(imageAccurate2_medium, imageAccurate1_medium, f_y * 120 + y, 1188, 396);
			}

			for (int y = 0; y != 120; ++y)
			{
				doHorizontalMiddle<5, &AccuratePixel::red>(imageAccurate2_medium, imageAccurate1_medium, f_y * 120 + y, 1584, 386);
				doHorizontalEnd<5, &AccuratePixel::red>(imageAccurate2_medium, imageAccurate1_medium, f_y * 120 + y);
				doHorizontalMiddle<5, &AccuratePixel::green>(imageAccurate2_medium, imageAccurate1_medium, f_y * 120 + y, 1584, 386);
				doHorizontalEnd<5, &AccuratePixel::green>(imageAccurate2_medium, imageAccurate1_medium, f_y * 120 + y);
				doHorizontalMiddle<5, &AccuratePixel::blue>(imageAccurate2_medium, imageAccurate1_medium, f_y * 120 + y, 1584, 386);
				doHorizontalEnd<5, &AccuratePixel::blue>(imageAccurate2_medium, imageAccurate1_medium, f_y * 120 + y);
			}
		}

		transpose(*imageAccurate1_medium, *imageAccurate2_medium);

		for (int f_y = 0; f_y != 10; ++f_y)
		{
			for (int y = 0; y != 198; ++y)
			{
				doHorizontalStart<5, &AccuratePixel::red>(imageAccurate2_medium, imageAccurate1_medium, f_y * 198 + y);
				doHorizontalMiddle<5, &AccuratePixel::red>(imageAccurate2_medium, imageAccurate1_medium, f_y * 198 + y, 0, 240);
				doHorizontalStart<5, &AccuratePixel::green>(imageAccurate2_medium, imageAccurate1_medium, f_y * 198 + y);
				doHorizontalMiddle<5, &AccuratePixel::green>(imageAccurate2_medium, imageAccurate1_medium, f_y * 198 + y, 0, 240);
				doHorizontalStart<5, &AccuratePixel::blue>(imageAccurate2_medium, imageAccurate1_medium, f_y * 198 + y);
				doHorizontalMiddle<5, &AccuratePixel::blue>(imageAccurate2_medium, imageAccurate1_medium, f_y * 198 + y, 0, 240);
			}

			for (int y = 0; y != 198; ++y)
			{
				doHorizontalMiddle<5, &AccuratePixel::red>(imageAccurate2_medium, imageAccurate1_medium, f_y * 198 + y, 240, 240);
				doHorizontalMiddle<5, &AccuratePixel::green>(imageAccurate2_medium, imageAccurate1_medium, f_y * 198 + y, 240, 240);
				doHorizontalMiddle<5, &AccuratePixel::blue>(imageAccurate2_medium, imageAccurate1_medium, f_y * 198 + y, 240, 240);
			}

			for (int y = 0; y != 198; ++y)
			{
				doHorizontalMiddle<5, &AccuratePixel::red>(imageAccurate2_medium, imageAccurate1_medium, f_y * 198 + y, 480, 240);
				doHorizontalMiddle<5, &AccuratePixel::green>(imageAccurate2_medium, imageAccurate1_medium, f_y * 198 + y, 480, 240);
				doHorizontalMiddle<5, &AccuratePixel::blue>(imageAccurate2_medium, imageAccurate1_medium, f_y * 198 + y, 480, 240);
			}

			for (int y = 0; y != 198; ++y)
			{
				doHorizontalMiddle<5, &AccuratePixel::red>(imageAccurate2_medium, imageAccurate1_medium, f_y * 198 + y, 720, 240);
				doHorizontalMiddle<5, &AccuratePixel::green>(imageAccurate2_medium, imageAccurate1_medium, f_y * 198 + y, 720, 240);
				doHorizontalMiddle<5, &AccuratePixel::blue>(imageAccurate2_medium, imageAccurate1_medium, f_y * 198 + y, 720, 240);
			}

			for (int y = 0; y != 198; ++y)
			{
				doHorizontalMiddle<5, &AccuratePixel::red>(imageAccurate2_medium, imageAccurate1_medium, f_y * 198 + y, 960, 240);
				doHorizontalMiddle<5, &AccuratePixel::green>(imageAccurate2_medium, imageAccurate1_medium, f_y * 198 + y, 960, 240);
				doHorizontalMiddle<5, &AccuratePixel::blue>(imageAccurate2_medium, imageAccurate1_medium, f_y * 198 + y, 960, 240);
			}

			for (int y = 0; y != 198; ++y)
			{
				doHorizontalMiddle<5, &AccuratePixel::red>(imageAccurate2_medium, imageAccurate1_medium, f_y * 198 + y, 960, 230);
				doHorizontalEnd<5, &AccuratePixel::red>(imageAccurate2_medium, imageAccurate1_medium, f_y * 198 + y);
				doHorizontalMiddle<5, &AccuratePixel::green>(imageAccurate2_medium, imageAccurate1_medium, f_y * 198 + y, 960, 230);
				doHorizontalEnd<5, &AccuratePixel::green>(imageAccurate2_medium, imageAccurate1_medium, f_y * 198 + y);
				doHorizontalMiddle<5, &AccuratePixel::blue>(imageAccurate2_medium, imageAccurate1_medium, f_y * 198 + y, 960, 230);
				doHorizontalEnd<5, &AccuratePixel::blue>(imageAccurate2_medium, imageAccurate1_medium, f_y * 198 + y);
			}
		}

		transpose(*imageAccurate1_medium, *imageAccurate2_medium);
	}

	AccurateImage *imageAccurate1_large = convertImageToNewFormat(image);
	AccurateImage *imageAccurate2_large = convertImageToNewFormat(image);

	for (int iterations = 0; iterations != 5; ++iterations)
	{
		for (int f_y = 0; f_y != 10; ++f_y)
		{
			for (int y = 0; y != 120; ++y)
			{
				doHorizontalStart<8, &AccuratePixel::red>(imageAccurate2_large, imageAccurate1_large, f_y * 120 + y);
				doHorizontalMiddle<8, &AccuratePixel::red>(imageAccurate2_large, imageAccurate1_large, f_y * 120 + y, 0, 396);
				doHorizontalStart<8, &AccuratePixel::green>(imageAccurate2_large, imageAccurate1_large, f_y * 120 + y);
				doHorizontalMiddle<8, &AccuratePixel::green>(imageAccurate2_large, imageAccurate1_large, f_y * 120 + y, 0, 396);
				doHorizontalStart<8, &AccuratePixel::blue>(imageAccurate2_large, imageAccurate1_large, f_y * 120 + y);
				doHorizontalMiddle<8, &AccuratePixel::blue>(imageAccurate2_large, imageAccurate1_large, f_y * 120 + y, 0, 396);
			}

			for (int y = 0; y != 120; ++y)
			{
				doHorizontalMiddle<8, &AccuratePixel::red>(imageAccurate2_large, imageAccurate1_large, f_y * 120 + y, 396, 396);
				doHorizontalMiddle<8, &AccuratePixel::green>(imageAccurate2_large, imageAccurate1_large, f_y * 120 + y, 396, 396);
				doHorizontalMiddle<8, &AccuratePixel::blue>(imageAccurate2_large, imageAccurate1_large, f_y * 120 + y, 396, 396);
			}

			for (int y = 0; y != 120; ++y)
			{
				doHorizontalMiddle<8, &AccuratePixel::red>(imageAccurate2_large, imageAccurate1_large, f_y * 120 + y, 792, 396);
				doHorizontalMiddle<8, &AccuratePixel::green>(imageAccurate2_large, imageAccurate1_large, f_y * 120 + y, 792, 396);
				doHorizontalMiddle<8, &AccuratePixel::blue>(imageAccurate2_large, imageAccurate1_large, f_y * 120 + y, 792, 396);
			}

			for (int y = 0; y != 120; ++y)
			{
				doHorizontalMiddle<8, &AccuratePixel::red>(imageAccurate2_large, imageAccurate1_large, f_y * 120 + y, 1188, 396);
				doHorizontalMiddle<8, &AccuratePixel::green>(imageAccurate2_large, imageAccurate1_large, f_y * 120 + y, 1188, 396);
				doHorizontalMiddle<8, &AccuratePixel::blue>(imageAccurate2_large, imageAccurate1_large, f_y * 120 + y, 1188, 396);
			}

			for (int y = 0; y != 120; ++y)
			{
				doHorizontalMiddle<8, &AccuratePixel::red>(imageAccurate2_large, imageAccurate1_large, f_y * 120 + y, 1584, 380);
				doHorizontalEnd<8, &AccuratePixel::red>(imageAccurate2_large, imageAccurate1_large, f_y * 120 + y);
				doHorizontalMiddle<8, &AccuratePixel::green>(imageAccurate2_large, imageAccurate1_large, f_y * 120 + y, 1584, 380);
				doHorizontalEnd<8, &AccuratePixel::green>(imageAccurate2_large, imageAccurate1_large, f_y * 120 + y);
				doHorizontalMiddle<8, &AccuratePixel::blue>(imageAccurate2_large, imageAccurate1_large, f_y * 120 + y, 1584, 380);
				doHorizontalEnd<8, &AccuratePixel::blue>(imageAccurate2_large, imageAccurate1_large, f_y * 120 + y);
			}
		}

		transpose(*imageAccurate1_large, *imageAccurate2_large);

		for (int f_y = 0; f_y != 10; ++f_y)
		{
			for (int y = 0; y != 198; ++y)
			{
				doHorizontalStart<8, &AccuratePixel::red>(imageAccurate2_large, imageAccurate1_large, f_y * 198 + y);
				doHorizontalMiddle<8, &AccuratePixel::red>(imageAccurate2_large, imageAccurate1_large, f_y * 198 + y, 0, 240);
				doHorizontalStart<8, &AccuratePixel::green>(imageAccurate2_large, imageAccurate1_large, f_y * 198 + y);
				doHorizontalMiddle<8, &AccuratePixel::green>(imageAccurate2_large, imageAccurate1_large, f_y * 198 + y, 0, 240);
				doHorizontalStart<8, &AccuratePixel::blue>(imageAccurate2_large, imageAccurate1_large, f_y * 198 + y);
				doHorizontalMiddle<8, &AccuratePixel::blue>(imageAccurate2_large, imageAccurate1_large, f_y * 198 + y, 0, 240);
			}

			for (int y = 0; y != 198; ++y)
			{
				doHorizontalMiddle<8, &AccuratePixel::red>(imageAccurate2_large, imageAccurate1_large, f_y * 198 + y, 240, 240);
				doHorizontalMiddle<8, &AccuratePixel::green>(imageAccurate2_large, imageAccurate1_large, f_y * 198 + y, 240, 240);
				doHorizontalMiddle<8, &AccuratePixel::blue>(imageAccurate2_large, imageAccurate1_large, f_y * 198 + y, 240, 240);
			}

			for (int y = 0; y != 198; ++y)
			{
				doHorizontalMiddle<8, &AccuratePixel::red>(imageAccurate2_large, imageAccurate1_large, f_y * 198 + y, 480, 240);
				doHorizontalMiddle<8, &AccuratePixel::green>(imageAccurate2_large, imageAccurate1_large, f_y * 198 + y, 480, 240);
				doHorizontalMiddle<8, &AccuratePixel::blue>(imageAccurate2_large, imageAccurate1_large, f_y * 198 + y, 480, 240);
			}

			for (int y = 0; y != 198; ++y)
			{
				doHorizontalMiddle<8, &AccuratePixel::red>(imageAccurate2_large, imageAccurate1_large, f_y * 198 + y, 720, 240);
				doHorizontalMiddle<8, &AccuratePixel::green>(imageAccurate2_large, imageAccurate1_large, f_y * 198 + y, 720, 240);
				doHorizontalMiddle<8, &AccuratePixel::blue>(imageAccurate2_large, imageAccurate1_large, f_y * 198 + y, 720, 240);
			}

			for (int y = 0; y != 198; ++y)
			{
				doHorizontalMiddle<8, &AccuratePixel::red>(imageAccurate2_large, imageAccurate1_large, f_y * 198 + y, 960, 240);
				doHorizontalMiddle<8, &AccuratePixel::green>(imageAccurate2_large, imageAccurate1_large, f_y * 198 + y, 960, 240);
				doHorizontalMiddle<8, &AccuratePixel::blue>(imageAccurate2_large, imageAccurate1_large, f_y * 198 + y, 960, 240);
			}

			for (int y = 0; y != 198; ++y)
			{
				doHorizontalMiddle<8, &AccuratePixel::red>(imageAccurate2_large, imageAccurate1_large, f_y * 198 + y, 960, 224);
				doHorizontalEnd<8, &AccuratePixel::red>(imageAccurate2_large, imageAccurate1_large, f_y * 198 + y);
				doHorizontalMiddle<8, &AccuratePixel::green>(imageAccurate2_large, imageAccurate1_large, f_y * 198 + y, 960, 224);
				doHorizontalEnd<8, &AccuratePixel::green>(imageAccurate2_large, imageAccurate1_large, f_y * 198 + y);
				doHorizontalMiddle<8, &AccuratePixel::blue>(imageAccurate2_large, imageAccurate1_large, f_y * 198 + y, 960, 224);
				doHorizontalEnd<8, &AccuratePixel::blue>(imageAccurate2_large, imageAccurate1_large, f_y * 198 + y);
			}
		}

		transpose(*imageAccurate1_large, *imageAccurate2_large);
	}

	//// Save the images.
	PPMImage *final_tiny = performNewIdeaFinalization(imageAccurate1_tiny, imageAccurate1_small);
	PPMImage *final_small = performNewIdeaFinalization(imageAccurate1_small, imageAccurate1_medium);
	PPMImage *final_medium = performNewIdeaFinalization(imageAccurate1_medium, imageAccurate1_large);

	if (argc > 1) {
		writePPM("flower_tiny.ppm", final_tiny);
		writePPM("flower_small.ppm", final_small);
		writePPM("flower_medium.ppm", final_medium);
	}
	else {
		writeStreamPPM(stdout, final_tiny);
		writeStreamPPM(stdout, final_small);
		writeStreamPPM(stdout, final_medium);
	}
	return 0;
}

