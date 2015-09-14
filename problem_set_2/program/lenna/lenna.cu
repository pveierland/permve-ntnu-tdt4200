#include <cstddef>
#include <iostream>
#include <stdexcept>
#include <string>

#include "lodepng.h"

struct image
{
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

//__global__
//void
//invert_pixel()
//{
//
//}

#define CUDA_CALL(f, ...) { \
    const cudaError_t status = f(__VA_ARGS__); \
    if (status) { throw std::runtime_error(std::string(#f " failed: ") + cudaGetErrorString(status)); } }

int
main(const int argc, const char* argv[])
{
    try
    {
        image image("lenna512x512_inv.png");

        std::cout << "width = " << image.width << " height = " << image.height << " size = " << image.data.size() << std::endl;

  //for ( int i = 0 ; i < image.width*image.height*3 ; i++ ) {
  //  image.data[i] = ~image.data[i];
  //}
        void* device_image;

        CUDA_CALL(cudaMalloc, &device_image, image.data.size());

        CUDA_CALL(cudaFree, device_image);

        //invert_pixel<<<1, 1>>>();
        
        //image.save("lenna512x512_orig.png");
    }
    catch (std::runtime_error& error)
    {
        std::cerr << error.what() << std::endl;
        return 1;
    }
}

