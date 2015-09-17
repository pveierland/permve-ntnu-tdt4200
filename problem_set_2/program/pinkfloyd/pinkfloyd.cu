#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <unistd.h>
#include "lodepng.h"


#include <cuda_runtime.h>

#include <cmath>
#include <fstream>
#include <iostream>
#include <ostream>
#include <stdexcept>
#include <string>
#include <vector>
#include <utility>

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


struct rgb
{
    float r;
    float g;
    float b;
};

struct hsv
{
    float h; // degrees
    float s;
    float v;
};

// http://stackoverflow.com/a/6930407
rgb
hsv2rgb(hsv in)
{
    float hh, p, q, t, ff;
    long  i;
    rgb   out;
    
    if (in.s <= 0.0) // < is bogus, just shuts up warnings
    {
        out.r = in.v;
        out.g = in.v;
        out.b = in.v;
        return out;
    }

    hh = in.h;

    if (hh >= 360.0) 
    {
        hh = 0.0;
    }

    hh /= 60.0;
    i = (long)hh;
    ff = hh - i;
    p = in.v * (1.0 - in.s);
    q = in.v * (1.0 - (in.s * ff));
    t = in.v * (1.0 - (in.s * (1.0 - ff)));
    
    switch(i)
    {
        case 0:
            out.r = in.v;
            out.g = t;
            out.b = p;
            break;
        case 1:
            out.r = q;
            out.g = in.v;
            out.b = p;
            break;
        case 2:
            out.r = p;
            out.g = in.v;
            out.b = t;
            break;
        case 3:
            out.r = p;
            out.g = q;
            out.b = in.v;
            break;
        case 4:
            out.r = t;
            out.g = p;
            out.b = in.v;
            break;
        case 5:
        default:
            out.r = in.v;
            out.g = p;
            out.b = q;
            break;
    }
    return out;     
}

struct line
{
    static
    line
    parse(const std::string& input)
    {
        line l;

        char buffer[64] = {};
        hsv color_hsv   = { 1.0f, 255.0f, 255.0f };

        const int elements = sscanf(
            input.c_str(),
            "line %f,%f %f,%f %f %f,%f%s",
            &l.x0, &l.y0, &l.x1, &l.y1,
            &l.thickness,
            &color_hsv.h, &color_hsv.v,
            buffer);

        // Optional saturation argument
        if (!((elements == 7) ||
              (elements == 8 && sscanf(buffer, ",%f", &color_hsv.s) == 1)))
        {
            throw std::runtime_error("invalid line element");
        }

        color_hsv.v /= 255.0f;
        color_hsv.s /= 255.0f;

        l.color = hsv2rgb(color_hsv);

        return l;
    }

    float x0, x1, y0, y1, thickness;
    rgb color;
};

struct vec3
{
    vec3() : x(), y(), z() { }
    vec3(float x, float y, float z) : x(x), y(y), z(z) { }

    float
    dot(const vec3& v) const
    {
        return x * v.x + y * v.y + z * v.z;
    }

    float x, y, z;
};

struct line_draw_info
{
    vec3 e0, e1, e2, e3;
    rgb color;
};

std::ostream&
operator<<(std::ostream& os, const rgb& color)
{
    os << "rgb{r: " << color.r
       << ", g: "   << color.g
       << ", b: "   << color.b
       << "}";
    
    return os;
}

std::ostream&
operator<<(std::ostream& os, const line& l)
{
    os << "line{x0: "     << l.x0
       << ", y0: "        << l.y0
       << ", x1: "        << l.x1
       << ", y1: "        << l.y1
       << ", thickness: " << l.thickness
       << ", color: "     << l.color
       << "}";

    return os;
}

struct geometry
{
    geometry(const std::string& filename)
    {
        std::ifstream input_file(filename.c_str());

        int line = 0;

        std::string input_file_line;
        while (std::getline(input_file, input_file_line))
        {
            if (line == 0)
            {
                if (sscanf(input_file_line.c_str(), "%d,%d", &width, &height) != 2)
                {
                    throw std::runtime_error("invalid width/height in input file");
                }
            }
            else if (input_file_line.find("line") == 0)
            {
                lines.push_back(line::parse(input_file_line));
            }
            ++line;
        }
    }

    int width, height;
    std::vector<line> lines;
};

std::ostream&
operator<<(std::ostream& os, const vec3& v)
{
    os << "vec3{x: " << v.x
       << ", y: "    << v.y
       << ", z: "    << v.z
       << "}";

    return os;
}

texture<float, 1, cudaReadModeElementType> gaussian_lut;
cudaArray* gaussian_lut_cuda_array;

__global__
void
woot(unsigned char* image,
     const line_draw_info* info, const int index,
     const int width, const int height)
{
    info = info + index;

    const int x = blockIdx.x * 25 + threadIdx.x;
    const int y = blockIdx.y * 25 + threadIdx.y;

    const float sx = (float)x / width;
    const float sy = (float)y / height;

    const float d0 = sx * info->e0.x + sy * info->e0.y + info->e0.z;
    const float d1 = sx * info->e1.x + sy * info->e1.y + info->e1.z;
    const float d2 = sx * info->e2.x + sy * info->e2.y + info->e2.z;
    const float d3 = sx * info->e3.x + sy * info->e3.y + info->e3.z;

    if (d0 >= 0.0f && d1 >= 0.0f && d2 >= 0.0f && d3 >= 0.0f)
    {
        unsigned char* pixel = image + 4 * (y * width + x);

        const float alpha = tex1D(gaussian_lut, d0 < d2 ? d0 : d2) *
                            tex1D(gaussian_lut, d1 < d3 ? d1 : d3);

        const float b_r = pixel[0] / 255.0f;
        const float b_g = pixel[1] / 255.0f;
        const float b_b = pixel[2] / 255.0f;
        const float b_a = pixel[3] / 255.0f;

        const float inverse_alpha = 1.0f - alpha;

        const float r = info->color.r * alpha + b_r * inverse_alpha;
        const float g = info->color.g * alpha + b_g * inverse_alpha;
        const float b = info->color.b * alpha + b_b * inverse_alpha;
        const float a = alpha + b_a * inverse_alpha;

        pixel[0] = static_cast<unsigned char>(255.0f * r);
        pixel[1] = static_cast<unsigned char>(255.0f * g),
        pixel[2] = static_cast<unsigned char>(255.0f * b);
        pixel[3] = static_cast<unsigned char>(255.0f * a);
    }
}

void
gaussian_lut_construct()
{
    float gaussian_filter[32];
    const int steps = sizeof(gaussian_filter) / sizeof(gaussian_filter[0]);

    // Calculate lookup table
    for (int i = 0; i != steps; ++i)
    {
        gaussian_filter[i] = std::exp(-std::pow(1.0 - i / 32.0f, 10) / 0.5);
    }

    // Allocate CUDA array in device memory
    cudaChannelFormatDesc channelDesc = 
        cudaCreateChannelDesc(steps, 0, 0, 0, cudaChannelFormatKindFloat);

    cuda_call(cudaMallocArray, &gaussian_lut_cuda_array, &channelDesc, steps);

    // Copy to device memory some data located at address h_data in host memory 
    cuda_call(cudaMemcpyToArray, gaussian_lut_cuda_array, 0, 0, gaussian_filter, sizeof(gaussian_filter), cudaMemcpyHostToDevice);
    
    // Set texture reference parameters
    gaussian_lut.addressMode[0] = cudaAddressModeClamp;
    gaussian_lut.filterMode     = cudaFilterModePoint;
    gaussian_lut.normalized     = true;
    
    // Bind the array to the texture reference
    cuda_call(cudaBindTextureToArray, gaussian_lut, gaussian_lut_cuda_array, channelDesc);
}

void
gaussian_lut_destroy()
{
    // Free device memory
    cuda_call(cudaFreeArray, gaussian_lut_cuda_array);
}

int
main(const int argc, const char* argv[])
{
    geometry g("floyd.txt");
    std::vector<line_draw_info> lines(g.lines.size());

    const std::size_t image_total_bytes = g.height * g.width * 4;

    gaussian_lut_construct();

    std::vector<unsigned char> image(image_total_bytes);

    for (int i = 3; i < image_total_bytes; i += 4)
    {
        image[i] = 255;
    }
    
    void* device_image_area;
    cuda_call(cudaMalloc, &device_image_area, image_total_bytes);
    
    cuda_call(cudaMemcpy,
              device_image_area, &image.front(),
              image_total_bytes, cudaMemcpyHostToDevice);

    void* device_lines;
    cuda_call(cudaMalloc, &device_lines, lines.size() * sizeof(line_draw_info));

    int i = 0;
    for (std::vector<line>::iterator line = g.lines.begin();
         line != g.lines.end();
         ++line, ++i)
    {
        line_draw_info& info = lines[i];

        const float r        = line->thickness / 10.0f;
        const float x_diff_a = line->x0 - line->x1;
        const float x_diff_b = line->x1 - line->x0;
        const float y_diff_a = line->y0 - line->y1;
        const float y_diff_b = line->y1 - line->y0;

        const float length = std::sqrt(std::abs(x_diff_a * x_diff_a - y_diff_a * y_diff_a));
        
        const float k = 2.0f / ((2.0f * r + line->thickness) * length);

        info.color = line->color;

        info.e0 = vec3(k * (line->y0 - line->y1),
                       k * (line->x1 - line->x0),
                       1 + k * (line->x0 * line->y1 - line->x1 * line->y0));

        info.e1 = vec3(k * (line->x1 - line->x0),
                       k * (line->y1 - line->y0),
                       1 + k * (line->x0 * line->x0 +
                                line->y0 * line->y0 -
                                line->x0 * line->x1 -
                                line->y0 * line->y1));

        info.e2 = vec3(k * (line->y1 - line->y0),
                       k * (line->x0 - line->x1),
                       1 + k * (line->x1 * line->y0 - line->x0 * line->y1));

        info.e3 = vec3(k * (line->x0 - line->x1),
                       k * (line->y0 - line->y1),
                       1 + k * (line->x1 * line->x1 +
                                line->y1 * line->y1 -
                                line->x0 * line->x1 -
                                line->y0 * line->y1));
    }

    //for (int x = 0; x != width; ++x)
    //{
    //    for (int y = 0; y != height; ++y)
    //    {
    //        unsigned char* pixel = &image.front() + 4 * (y * width + x);
    //        pixel[0] = static_cast<unsigned char>(std::min(255.0f, 255.0f * (float)pixel[0] / pixel[3]));
    //        pixel[1] = static_cast<unsigned char>(std::min(255.0f, 255.0f * (float)pixel[1] / pixel[3]));
    //        pixel[2] = static_cast<unsigned char>(std::min(255.0f, 255.0f * (float)pixel[2] / pixel[3]));
    //        pixel[3] = static_cast<unsigned char>(std::min(255.0f, 255.0f * (float)pixel[3] / pixel[3]));
    //    }
    //}

    cuda_call(cudaMemcpy,
              device_lines, &lines.front(),
              lines.size() * sizeof(line_draw_info), cudaMemcpyHostToDevice);

    for (int i = 0; i != lines.size(); ++i)
    {
        dim3 gridSize(12, 12);
        dim3 blockSize(25,25);

        cuda_launch(woot, gridSize, blockSize,
            (unsigned char*)device_image_area,
            (const line_draw_info*)device_lines,
            i,
            g.width, g.height);

        cuda_call(cudaDeviceSynchronize);
    }

    cuda_call(cudaMemcpy,
              &image.front(), device_image_area,
              image_total_bytes, cudaMemcpyDeviceToHost);

    cuda_call(cudaFree, device_lines);
    cuda_call(cudaFree, device_image_area);
    gaussian_lut_destroy();

    lodepng::encode("floyd.png", image, g.width, g.height);
}

