#include <iostream>
#define _USE_MATH_DEFINES
#include <cmath>
#include "image.hpp"

Image make_gx_filter()
{
    Image filter(3, 3, 1);
    float values[9] = {
        -1, 0, 1,
        -2, 0, 2,
        -1, 0, 1
    };
    for (int i = 0; i < 9; i++) {
        filter.data[i] = values[i];
    }
    return filter;
}

Image make_gy_filter()
{
    Image filter(3, 3, 1);
    float values[9] = {
        -1, -2, -1,
        0, 0, 0,
        1, 2, 1
    };
    for (int i = 0; i < 9; i++) {
        filter.data[i] = values[i];
    }
    return filter;
}

Image make_gaussian_filter(float sigma)
{
    int size = std::ceil(6 * sigma);
    if (size % 2 == 0)
        size++;
    int center = size / 2;
    Image filter(size, size, 1);
    for (int x = -size/2; x <= size/2; x++) {
        for (int y = -size/2; y <= size/2; y++) {
            float val = 1 / (2*M_PI*sigma*sigma) * std::exp(-(x*x + y*y) / (2*sigma*sigma));
            filter.set_pixel(center+x, center+y, 0, val);
        }
    }
    float sum;
    for (int i = 0; i < size*size; i++)
        sum += filter.data[i];
    for (int i = 0; i < size*size; i++)
        filter.data[i] /= sum;
    return filter;
}
