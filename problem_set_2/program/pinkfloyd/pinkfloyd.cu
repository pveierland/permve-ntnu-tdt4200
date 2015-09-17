#include "lodepng.h"
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <ostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#define cuda_call(f, ...) \
    cuda_assert(f(__VA_ARGS__), __FILE__, __LINE__, #f)

#define cuda_launch(kernel, grid_dim, block_dim, ...) \
{ \
    kernel<<<grid_dim, block_dim>>>(__VA_ARGS__); \
    cuda_assert(cudaPeekAtLastError(), __FILE__, __LINE__, "kernel " #kernel " launch"); \
}

#define cuda_launch_stream(kernel, grid_dim, block_dim, stream, ...) \
{ \
    kernel<<<grid_dim, block_dim, 0, stream>>>(__VA_ARGS__); \
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

struct vec3
{
    float x, y, z;
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

struct line
{
    static
    line
    parse(const std::string& input)
    {
        line l;

        char buffer[64] = {};
        hsv color_hsv   = { 1.0f, 255.0f, 255.0f };

        const int elements = std::sscanf(
            input.c_str(),
            "line %f,%f %f,%f %f %f,%f%s",
            &l.x0, &l.y0, &l.x1, &l.y1,
            &l.thickness,
            &color_hsv.h, &color_hsv.v,
            buffer);

        // Optional saturation argument
        if (!((elements == 7) ||
              (elements == 8 && std::sscanf(buffer, ",%f", &color_hsv.s) == 1)))
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
    vec3 e0, e1, e2, e3;
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

        if (!input_file)
        {
            throw std::runtime_error("failed to open input file: " + filename);
        }

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

const std::size_t image_chunk_size = 64;

texture<float, 1, cudaReadModeElementType> gaussian_lut;
cudaArray* gaussian_lut_cuda_array;

__global__
void
calculate_line_equations(line* const lines)
{
    line* const line = lines + threadIdx.x;

    const float r        = line->thickness / 10.0f;
    const float x_diff_a = line->x0 - line->x1;
    const float x_diff_b = line->x1 - line->x0;
    const float y_diff_a = line->y0 - line->y1;
    const float y_diff_b = line->y1 - line->y0;

    const float length = std::sqrt(std::abs(x_diff_a * x_diff_a - y_diff_a * y_diff_a));
    
    const float k = 2.0f / ((2.0f * r + line->thickness) * length);

    const float x0x0 = line->x0 * line->x0;
    const float x0x1 = line->x0 * line->x1;
    const float x0y1 = line->x0 * line->y1;
    const float x1x1 = line->x1 * line->x1;
    const float x1y0 = line->x1 * line->y0;
    const float y0y0 = line->y0 * line->y0;
    const float y0y1 = line->y0 * line->y1;
    const float y1y1 = line->y1 * line->y1;

    line->e0.x = k * y_diff_a;
    line->e0.y = k * x_diff_b;
    line->e0.z = 1.0f + k * (x0y1 - x1y0);

    line->e1.x = k * x_diff_b;
    line->e1.y = k * y_diff_b;
    line->e1.z = 1.0f + k * (x0x0 + y0y0 - x0x1 - y0y1);

    line->e2.x = k * y_diff_b;
    line->e2.y = k * x_diff_a;
    line->e2.z = 1.0f + k * (x1y0 - x0y1);

    line->e3.x = k * x_diff_a;
    line->e3.y = k * y_diff_a;
    line->e3.z = 1.0f + k * (x1x1 + y1y1 - x0x1 - y0y1);
}

__global__
void
draw_line(unsigned char* const image,
          const line* const    line,
          const int            stream,
          const int            width,
          const int            height)
{
    const int index = stream * image_chunk_size * image_chunk_size +
                      blockIdx.x * image_chunk_size +
                      threadIdx.x;

    const int y = index / width;
    const int x = index - y * width;

    if (x < width && y < height)
    {
        const float sx = (float)x / width;
        const float sy = (float)y / height;

        const float d0 = sx * line->e0.x + sy * line->e0.y + line->e0.z;
        const float d1 = sx * line->e1.x + sy * line->e1.y + line->e1.z;
        const float d2 = sx * line->e2.x + sy * line->e2.y + line->e2.z;
        const float d3 = sx * line->e3.x + sy * line->e3.y + line->e3.z;

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

            const float r = line->color.r * alpha + b_r * inverse_alpha;
            const float g = line->color.g * alpha + b_g * inverse_alpha;
            const float b = line->color.b * alpha + b_b * inverse_alpha;
            const float a = alpha + b_a * inverse_alpha;

            pixel[0] = static_cast<unsigned char>(255.0f * r);
            pixel[1] = static_cast<unsigned char>(255.0f * g),
            pixel[2] = static_cast<unsigned char>(255.0f * b);
            pixel[3] = static_cast<unsigned char>(255.0f * a);
        }
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
    try
    {
        geometry g("input_tdsotm.txt");
    
        const std::size_t image_total_bytes = g.height * g.width * 4;
    
        gaussian_lut_construct();
    
        std::vector<unsigned char> image(image_total_bytes);
    
        // Initialize image alpha values
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
        cuda_call(cudaMalloc, &device_lines, g.lines.size() * sizeof(line));
    
        cuda_call(cudaMemcpy,
                  device_lines, &g.lines.front(),
                  g.lines.size() * sizeof(line), cudaMemcpyHostToDevice);
        
        cuda_launch(calculate_line_equations, 1, g.lines.size(), (line*)device_lines);
        
        const std::size_t number_of_streams =
            (g.width * g.height + image_chunk_size * image_chunk_size - 1) /
            (image_chunk_size * image_chunk_size);
    
        std::vector<cudaStream_t> streams(number_of_streams);
    
        for (int s = 0; s != number_of_streams; ++s)
        {
            cuda_call(cudaStreamCreate, &streams[s]);
        }
    
        for (int l = 0; l != g.lines.size(); ++l)
        {
            for (int s = 0; s != number_of_streams; ++s)
            {
                cuda_launch_stream(draw_line,
                                   image_chunk_size,
                                   image_chunk_size,
                                   streams[s],
                                   (unsigned char*)device_image_area,
                                   (const line*)device_lines + l,
                                   s,
                                   g.width,
                                   g.height);
            }
        }
    
        cuda_call(cudaMemcpy,
                  &image.front(), device_image_area,
                  image_total_bytes, cudaMemcpyDeviceToHost);
    
        for (int s = 0; s != number_of_streams; ++s)
        {
            cuda_call(cudaStreamDestroy, streams[s]);
        }
    
        cuda_call(cudaFree, device_lines);
        cuda_call(cudaFree, device_image_area);
        gaussian_lut_destroy();
    
        lodepng::encode("floyd.png", image, g.width, g.height);
    }
    catch (const std::runtime_error& e)
    {
        std::cerr << e.what() << std::endl;
        return 1;
    }
}

