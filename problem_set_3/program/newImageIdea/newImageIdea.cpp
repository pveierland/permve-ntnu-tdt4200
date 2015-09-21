#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

#include <cmath>
#include <iostream>
#include <vector>

#include "ppm.h"

// Image from:
// http://7-themes.com/6971875-funny-flowers-pictures.html

typedef struct {
    float red,green,blue;
} AccuratePixel;

typedef struct {
     int x, y;
     AccuratePixel *data;
} AccurateImage;

struct IntegralImage
{
    using IntegralType = long;
    using ResultType = double;

    IntegralImage(PPMImage& image)
        : width(image.x), height(image.y)
    {
        const std::size_t numPixels = image.x * image.y;

        integral.resize(3 * numPixels);
        output.resize(3 * numPixels);

        unsigned char* input = &image.data->red;

        integrate(input + 0, integral.data());
        integrate(input + 1, integral.data() + numPixels);
        integrate(input + 2, integral.data() + 2 * numPixels);
    }

    void
    integrate(unsigned char* input, IntegralType* output)
    {
        IntegralType *outputLastRow, *outputThisRow;
        outputLastRow = outputThisRow = output;

        IntegralType rowSum{};
        for (int x = width; x != 0; --x)
        {
            rowSum += *input;
            input += 3;
            *outputThisRow++ = rowSum;
        }

        for (int y = height; y != 1; --y)
        {
            rowSum = IntegralType{};
            for (int x = width; x != 0; --x)
            {
                rowSum += *input;
                input += 3;
                *outputThisRow++ = rowSum + *outputLastRow++;
            }
        }
    }
    
    void
    reintegrate(ResultType* input, IntegralType* output)
    {
        IntegralType *outputLastRow, *outputThisRow;
        outputLastRow = outputThisRow = output;

        IntegralType rowSum{};
        for (int x = width; x != 0; --x)
        {
            rowSum += *input++;
            *outputThisRow++ = rowSum;
        }

        for (int y = height; y != 1; --y)
        {
            rowSum = IntegralType{};
            for (int x = width; x != 0; --x)
            {
                rowSum += *input++;
                *outputThisRow++ = rowSum + *outputLastRow++;
            }
        }
    }

    inline
    IntegralType sum(IntegralType* ul, IntegralType* ur, IntegralType* ll, IntegralType* lr)
    {
        return *lr - *ll - *ur + *ul;
    }

    void
    iterate(int size, int iterations)
    {
        while (true)
        {
            const std::size_t numPixels = width * height;

            filter(integral.data(), output.data(), size);
            filter(integral.data() + numPixels, output.data() + numPixels, size);
            filter(integral.data() + 2 * numPixels, output.data() + 2 * numPixels, size);

            if (--iterations == 0)
            {
                break;
            } 

            reintegrate(output.data(), integral.data());
            reintegrate(output.data() + numPixels, integral.data() + numPixels);
            reintegrate(output.data() + 2 * numPixels, integral.data() + 2 * numPixels);
        }
    }

    void
    filter(IntegralType* input, ResultType* output, int size)
    {
        int N = (size + 1) * (size + 1);
        int s = size;

        IntegralType* lr = input + width * size + size;

        // DO UNTIL OUT OF STARTZONE

        for (int i = size; i != 0; --i)
        {
            *output++ = *lr++ / N;
            N += size + 1;
        }

        *output++ = *lr++ / N;

        IntegralType* ll = input + width * size;

        for (int i = width - 2 * size - 2; i != 0; --i)
        {
            *output++ = (*lr++ - *ll++) / N;
        }

        *output++ = (*lr - *ll++) / N;

        for (int i = size; i != 0; --i)
        {
            N -= size + 1;
            *output++ = (*lr - *ll++) / N;
        }
    }

    AccurateImage
    accurate()
    {
        const std::size_t numPixels = width * height;

        AccuratePixel* data = (AccuratePixel*)malloc(sizeof(AccuratePixel) * numPixels);
        AccurateImage img{width, height, data};

        auto red   = output.data();
        auto green = output.data() + numPixels;
        auto blue  = output.data() + 2 * numPixels;

        for (int i = 0; i != numPixels; ++i)
        {
            data->red = *red++;
            data->green = *green++;
            data->blue = *blue++;
            ++data;
        }

        return img;
    }

    int width, height;
    std::vector<IntegralType> integral;
    std::vector<ResultType> output;
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

void performNewIdeaIterationR(AccurateImage *imageOut, AccurateImage *imageIn, int size) {
	
	// Iterate over each pixel
	for(int senterX = 0; senterX < imageIn->x; senterX++) {
	
		for(int senterY = 0; senterY < imageIn->y; senterY++) {
			
			// For each pixel we compute the magic number
			float sum = 0;
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
				    sum += imageIn->data[offsetOfThePixel].red;

                    if (senterX == 896 && senterY == 232)
                    {
                        std::cout << "SUMMING: " <<imageIn->data[offsetOfThePixel].red << std::endl;
                    }
					
					// Keep track of how many values we have included
					countIncluded++;
				}
			
			}
			
			// Now we compute the final value
			float value = sum / countIncluded;
			
			
			// Update the output image
			int numberOfValuesInEachRow = imageOut->x; // R, G and B
			int offsetOfThePixel = (numberOfValuesInEachRow * senterY + senterX);
			imageOut->data[offsetOfThePixel].red = value;
		}
	
	}
	
}

void performNewIdeaIterationG(AccurateImage *imageOut, AccurateImage *imageIn, int size) {
	
	// Iterate over each pixel
	for(int senterX = 0; senterX < imageIn->x; senterX++) {
	
		for(int senterY = 0; senterY < imageIn->y; senterY++) {
			
			// For each pixel we compute the magic number
			float sum = 0;
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
					sum += imageIn->data[offsetOfThePixel].green;
					
					// Keep track of how many values we have included
					countIncluded++;
				}
			
			}
			
			// Now we compute the final value
			float value = sum / countIncluded;
			
			
			// Update the output image
			int numberOfValuesInEachRow = imageOut->x; // R, G and B
			int offsetOfThePixel = (numberOfValuesInEachRow * senterY + senterX);
			imageOut->data[offsetOfThePixel].green = value;
		}
	
	}
	
}

void performNewIdeaIterationB(AccurateImage *imageOut, AccurateImage *imageIn, int size) {
	
	// Iterate over each pixel
	for(int senterX = 0; senterX < imageIn->x; senterX++) {
	
		for(int senterY = 0; senterY < imageIn->y; senterY++) {
			
			// For each pixel we compute the magic number
			float sum = 0;
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
					
                    sum += imageIn->data[offsetOfThePixel].blue;
					
					// Keep track of how many values we have included
					countIncluded++;
				}
			
			}
			
			// Now we compute the final value
			float value = sum / countIncluded;
			
			
			// Update the output image
			int numberOfValuesInEachRow = imageOut->x; // R, G and B
			int offsetOfThePixel = (numberOfValuesInEachRow * senterY + senterX);
			imageOut->data[offsetOfThePixel].blue = value;
		}
	
	}
	
}

// Perform the new idea:
void performNewIdeaIteration(AccurateImage *imageOut, AccurateImage *imageIn, int colourType, int size) {
    switch (colourType)
    {
        case 0: performNewIdeaIterationR(imageOut, imageIn, size); break;
        case 1: performNewIdeaIterationG(imageOut, imageIn, size); break;
        case 2: performNewIdeaIterationB(imageOut, imageIn, size); break;
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


int main(int argc, char** argv) {
	PPMImage *image;
	
	if(argc > 1) {
		image = readPPM("flower.ppm");
	} else {
		image = readStreamPPM(stdin);
	}

    IntegralImage ii{*image};

   int special_x = 896;
   int special_y = 232;

//    std::cout << "LR: " << ii.integral[(special_y + 1) * ii.width + special_x + 1]
//              << "\nLL: " << ii.integral[(special_y + 1) * ii.width + special_x - 2]
//              << "\nUR: " << ii.integral[(special_y - 2) * ii.width + special_x + 1]
//              << "\nUL: " << ii.integral[(special_y - 2) * ii.width + special_x - 2] << std::endl;
    
	    AccurateImage *imageAccurate1_tiny = convertImageToNewFormat(image);
	    AccurateImage *imageAccurate2_tiny = convertImageToNewFormat(image);

    AccurateImage tiny = ii.accurate();
        AccurateImage *alpha = &tiny, *bravo = imageAccurate2_tiny;
        
//   for (int y = special_y - 2; y != special_y + 2; ++y)
//    {
//        for (int x = special_x - 2; x != special_x + 2; ++x)
//        {
//            std::cout << "y: " << y << " x: " << x << " a: " << alpha->data[y * alpha->x + x].red << " b: " << bravo->data[y * alpha->x + x].red << std::endl; 
//        }
//    }

   std::cout << std::endl;
   



    ii.iterate(1, 2);
    tiny = ii.accurate();

    {
	    int size = 2; 
	    performNewIdeaIteration(imageAccurate2_tiny, imageAccurate1_tiny, 0, size);
	    performNewIdeaIteration(imageAccurate2_tiny, imageAccurate1_tiny, 1 , size);
	    performNewIdeaIteration(imageAccurate2_tiny, imageAccurate1_tiny, 2, size);
    }

    //for (int y = 0; y != alpha->y; ++y)
    //{
    //    for (int x = 0; x != alpha->x; ++x)
    //    {
    //        float v = alpha->data[y * alpha->x + x].red - bravo->data[y * alpha->x + x].red;
    //        if (std::fabs(v) > 10.0f)
    //        {
    //            std::cout << "y: " << y << " x: " << x << " a: " << alpha->data[y * alpha->x + x].red << " b: " << bravo->data[y * alpha->x + x].red << " diff: " << std::fabs(v) << std::endl; 
    //        }
    //    }
    //}
//   for (int y = special_y - 2; y != special_y + 2; ++y)
//    {
//        for (int x = special_x - 2; x != special_x + 2; ++x)
//        {
//            std::cout << "y: " << y << " x: " << x << " a: " << alpha->data[y * alpha->x + x].red << " b: " << bravo->data[y * alpha->x + x].red << std::endl; 
//        }
//    }
        for (int x = 0; x != ii.width / 10; ++x)
        {
            std::cout << " x: " << x << " a: " << alpha->data[x].red << " b: " << bravo->data[x].red << std::endl; 
        }
    std::cout << std::endl;

    std::cout << "LR: " << ii.integral[(special_y + 1) * ii.width + special_x + 1]
              << "\nLL: " << ii.integral[(special_y + 1) * ii.width + special_x - 2]
              << "\nUR: " << ii.integral[(special_y - 2) * ii.width + special_x + 1]
              << "\nUL: " << ii.integral[(special_y - 2) * ii.width + special_x - 2] << std::endl;


    return 0;

    //for (int y = 0; y != 3; ++y)
    //{
    //    for (int x = 0; x != 3; ++x)
    //    {
    //        std::cout << tiny.data[tiny.x * y + x].red << " ";
    //    }
    //    std::cout << std::endl;
    //}

    //std::cout << std::endl;
    //for (int y = 0; y != 3; ++y)
    //{
    //    for (int x = 0; x != 3; ++x)
    //    {
    //        std::cout << imageAccurate2_tiny->data[tiny.x * y + x].red << " ";
    //    }
    //    std::cout << std::endl;
    //}


	//
	//// Process the tiny case:
	//for(int colour = 0; colour < 3; colour++) {
	//	int size = 2; 
	//	performNewIdeaIteration(imageAccurate2_tiny, imageAccurate1_tiny, colour, size);
	//	performNewIdeaIteration(imageAccurate1_tiny, imageAccurate2_tiny, colour, size);
	//	performNewIdeaIteration(imageAccurate2_tiny, imageAccurate1_tiny, colour, size);
	//	performNewIdeaIteration(imageAccurate1_tiny, imageAccurate2_tiny, colour, size);
	//	performNewIdeaIteration(imageAccurate2_tiny, imageAccurate1_tiny, colour, size);
	//}
	
	
	AccurateImage *imageAccurate1_small = convertImageToNewFormat(image);
	AccurateImage *imageAccurate2_small = convertImageToNewFormat(image);
	
	// Process the small case:
	for(int colour = 0; colour < 3; colour++) {
		int size = 3;
		performNewIdeaIteration(imageAccurate2_small, imageAccurate1_small, colour, size);
		performNewIdeaIteration(imageAccurate1_small, imageAccurate2_small, colour, size);
		performNewIdeaIteration(imageAccurate2_small, imageAccurate1_small, colour, size);
		performNewIdeaIteration(imageAccurate1_small, imageAccurate2_small, colour, size);
		performNewIdeaIteration(imageAccurate2_small, imageAccurate1_small, colour, size);
	}
	
	AccurateImage *imageAccurate1_medium = convertImageToNewFormat(image);
	AccurateImage *imageAccurate2_medium = convertImageToNewFormat(image);
	
	// Process the medium case:
	for(int colour = 0; colour < 3; colour++) {
		int size = 5;
		performNewIdeaIteration(imageAccurate2_medium, imageAccurate1_medium, colour, size);
		performNewIdeaIteration(imageAccurate1_medium, imageAccurate2_medium, colour, size);
		performNewIdeaIteration(imageAccurate2_medium, imageAccurate1_medium, colour, size);
		performNewIdeaIteration(imageAccurate1_medium, imageAccurate2_medium, colour, size);
		performNewIdeaIteration(imageAccurate2_medium, imageAccurate1_medium, colour, size);
	}
	
	AccurateImage *imageAccurate1_large = convertImageToNewFormat(image);
	AccurateImage *imageAccurate2_large = convertImageToNewFormat(image);
	
	// Do each color channel
	for(int colour = 0; colour < 3; colour++) {
		int size = 8;
		performNewIdeaIteration(imageAccurate2_large, imageAccurate1_large, colour, size);
		performNewIdeaIteration(imageAccurate1_large, imageAccurate2_large, colour, size);
		performNewIdeaIteration(imageAccurate2_large, imageAccurate1_large, colour, size);
		performNewIdeaIteration(imageAccurate1_large, imageAccurate2_large, colour, size);
		performNewIdeaIteration(imageAccurate2_large, imageAccurate1_large, colour, size);
	}
	
	// Save the images.
	PPMImage *final_tiny = performNewIdeaFinalization(imageAccurate2_tiny,  imageAccurate2_small);
	PPMImage *final_small = performNewIdeaFinalization(imageAccurate2_small,  imageAccurate2_medium);
	PPMImage *final_medium = performNewIdeaFinalization(imageAccurate2_medium,  imageAccurate2_large);
	
	if(argc > 1) {
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

