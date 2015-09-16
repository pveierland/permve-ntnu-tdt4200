#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <unistd.h>
#include "lodepng.h"



#include <cmath>
#include <fstream>
#include <iostream>
#include <ostream>
#include <stdexcept>
#include <string>
#include <vector>
#include <utility>

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

        // Optional HSV saturation argument
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

        std::string input_file_line;
        while (std::getline(input_file, input_file_line))
        {
            if (input_file_line.find("line") == 0)
            {
                lines.push_back(line::parse(input_file_line));
            }
        }
    }

    std::vector<line> lines;
};

struct vec3
{
    vec3(float x, float y, float z) : x(x), y(y), z(z) { }

    float
    dot(const vec3& v) const
    {
        return x * v.x + y * v.y + z * v.z;
    }

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

/*
struct vec3{
	float x,y,z;
};

void drawFigures(struct LineInfo * gpulines, const int lines, struct CircleInfo * gpucircles, const int circles, unsigned char * image, int height, int width, float*floatimg, struct vec3 blockIdx, struct vec3 threadIdx){
}




void  parseCircle(char * line, struct CircleInfo ci[], size_t *circles){
  float x,y,radius;
  struct Color c;
  int items = sscanf(line, "circle %f,%f %f %f,%f", &x,&y,&radius, &c.angle, &c.intensity);
  if ( 5==items){
    ci[*circles].x = x;
    ci[*circles].y = y;
    ci[*circles].radius = radius;
    ci[*circles].color.angle = c.angle;
    ci[*circles].color.intensity = c.intensity;
    (*circles)++;
  }
}


void printLines(struct LineInfo li[], size_t lines){
  for ( int i = 0 ; i < lines ; i++){
    printf("line:  from:%f,%f to:%f,%f thick:%f,  %f,%f\n", li[i].x0, li[i].y0, li[i].x1, li[i].y1, li[i].thickness,li[i].color.angle, li[i].color.intensity);
  }
}

void printCircles(struct CircleInfo ci[], size_t circles){
  for ( int i = 0 ; i < circles ; i++){
    printf("circle %f,%f %f %f,%f\n", ci[i].x,ci[i].y,ci[i].radius, ci[i].color.angle, ci[i].color.intensity);
  }
}

*/


float
gaz(float x)
{
    return std::exp(-std::pow(1.0 - x, 10) / 0.5);
}


float
goat(float x)
{
    return x < 1.0 ? gaz(x) : 1.0;
}

void
combine(unsigned char* pixel,
        float a_r,
        float a_g,
        float a_b,
        float a_a)
{
    float b_r = pixel[0] / 255.0f;
    float b_g = pixel[1] / 255.0f;
    float b_b = pixel[2] / 255.0f;
    float b_a = pixel[3] / 255.0f;

    float r = a_r + b_r * (1.0 - a_a);
    float g = a_g + b_g * (1.0 - a_a);
    float b = a_b + b_b * (1.0 - a_a);
    float a = a_a + b_a * (1.0 - a_a);

    pixel[0] = static_cast<unsigned char>(std::min(255.0f, 255.0f * r));
    pixel[1] = static_cast<unsigned char>(std::min(255.0f, 255.0f * g)),
    pixel[2] = static_cast<unsigned char>(std::min(255.0f, 255.0f * b));
    pixel[3] = static_cast<unsigned char>(std::min(255.0f, 255.0f * a));
}

void
blend  (unsigned char* pixel,
        float a_r,
        float a_g,
        float a_b,
        float a_a)
{
    float b_r = pixel[0] / 255.0f;
    float b_g = pixel[1] / 255.0f;
    float b_b = pixel[2] / 255.0f;
    float b_a = pixel[3] / 255.0f;

    float a = std::min(1.0f, a_a + b_a);

    float r = (a_r + b_r) / a;
    float g = (a_g + b_g) / a;
    float b = (a_b + b_b) / a;

    pixel[0] = static_cast<unsigned char>(std::min(255.0f, 255.0f * r));
    pixel[1] = static_cast<unsigned char>(std::min(255.0f, 255.0f * g)),
    pixel[2] = static_cast<unsigned char>(std::min(255.0f, 255.0f * b));
    pixel[3] = static_cast<unsigned char>(std::min(255.0f, 255.0f * a));
}

int
main(const int argc, const char* argv[])
{
    const int width  = 300;
    const int height = 300;

    geometry g("floyd.txt");

    std::vector<unsigned char> image(width * height * 4, 0);

    for (int x = 0; x != width; ++x)
    {
        for (int y = 0; y != height; ++y)
        {
            unsigned char* pixel = &image.front() + 4 * (y * width + x);
            pixel[0] = 0;
            pixel[1] = 0;
            pixel[2] = 0;
            pixel[3] = 255;
        }
    }

    float m1 = 0.0f, m2 = 0.0f;

    for (std::vector<line>::iterator line = g.lines.begin();
         line != g.lines.end();
         ++line)
    {
        std::cout << *line << std::endl;

        std::swap(line->x0, line->x1);
        std::swap(line->y0, line->y1);

        const float r        = line->thickness / 10.0f;
        const float x_diff_a = line->x0 - line->x1;
        const float x_diff_b = line->x1 - line->x0;
        const float y_diff_a = line->y0 - line->y1;
        const float y_diff_b = line->y1 - line->y0;

        const float length = std::sqrt(std::abs(x_diff_a * x_diff_a - y_diff_a * y_diff_a));

        const float k = 2.0f / ((2.0f * r + line->thickness) * length);

        const vec3 e0(k * (line->y0 - line->y1),
                      k * (line->x1 - line->x0),
                      1 + k * (line->x0 * line->y1 - line->x1 * line->y0));
        const vec3 e1(k * (line->x1 - line->x0),
                      k * (line->y1 - line->y0),
                      1 + k * (line->x0 * line->x0 +
                               line->y0 * line->y0 -
                               line->x0 * line->x1 -
                               line->y0 * line->y1));
        const vec3 e2(k * (line->y1 - line->y0),
                      k * (line->x0 - line->x1),
                      1 + k * (line->x1 * line->y0 - line->x0 * line->y1));
        const vec3 e3(k * (line->x0 - line->x1),
                      k * (line->y0 - line->y1),
                      1 + k * (line->x1 * line->x1 +
                               line->y1 * line->y1 -
                               line->x0 * line->x1 -
                               line->y0 * line->y1));

        for (int x = 0; x != width; ++x)
        {
            for (int y = 0; y != height; ++y)
            {
                const float sx = (float)x / width;
                const float sy = (float)y / height;

                const vec3 v(sx, sy, 1);

                const float d0 = v.dot(e0);
                const float d1 = v.dot(e1);
                const float d2 = v.dot(e2);
                const float d3 = v.dot(e3);

                if (d0 >= 0.0f && d1 >= 0.0f && d2 >= 0.0f && d3 >= 0.0f)
                {
                    m1 = std::max(m1, std::min(d0, d2));
                    m2 = std::max(m2, std::min(d1, d3));
                    unsigned char* pixel = &image.front() + 4 * (y * width + x);
                    
                    const float alpha = gaz(std::min(d0, d2)) * goat(std::min(d1, d3));

                    combine(pixel,
                            alpha * line->color.r, 
                            alpha * line->color.g,
                            alpha * line->color.b,
                            alpha);

                    //pixel[0] = static_cast<unsigned char>(255.0f * (r + pixel[0] / 255.0f * (1.0f - r)));
                    //pixel[1] = static_cast<unsigned char>(255.0f * (g + pixel[1] / 255.0f * (1.0f - g)));
                    //pixel[2] = static_cast<unsigned char>(255.0f * (b + pixel[2] / 255.0f * (1.0f - b)));
                    //pixel[3] = static_cast<unsigned char>(255.0f * (intensity + pixel[3] / 255.0f * (1.0f - intensity)));
                }
            }
        }
    }

//    for (int x = 0; x != width; ++x)
//    {
//        for (int y = 0; y != height; ++y)
//        {
//            unsigned char* pixel = &image.front() + 4 * (y * width + x);
//            pixel[0] = static_cast<unsigned char>(std::min(255.0f, 255.0f * (float)pixel[0] / pixel[3]));
//            pixel[1] = static_cast<unsigned char>(std::min(255.0f, 255.0f * (float)pixel[1] / pixel[3]));
//            pixel[2] = static_cast<unsigned char>(std::min(255.0f, 255.0f * (float)pixel[2] / pixel[3]));
//            pixel[3] = static_cast<unsigned char>(std::min(255.0f, 255.0f * (float)pixel[3] / pixel[3]));
//        }
//    }

    std::cout << "m1 = " << m1 << ", m2 = " << m2 << std::endl;

    lodepng::encode("floyd.png", image, width, height);
}

