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


Image convolve(const Image& img, const Image& filter, bool preserve)
{
    assert(filter.channels == img.channels || filter.channels == 1);
    int new_c = 1;
    if (preserve) //preserve number of channels after filtering
        new_c = img.channels;
    Image filtered(img.width, img.height, new_c);
    for (int x = 0; x < img.width; x++) {
        for (int y = 0; y < img.height; y++) {
            for (int c = 0; c < img.channels; c++) {
                float val = 0;
                for (int i = 0; i < filter.width; i++) {
                    for (int j = 0; j < filter.height; j++) {
                        int dx = -filter.width/2 + i;
                        int dy = -filter.height/2 + j;
                        int filter_c = (filter.channels == 1) ? 0 : c;
                        val += img.get_pixel(x+dx, y+dy, c) * filter.get_pixel(i, j, filter_c);
                    }
                }
                if (preserve) {
                    filtered.set_pixel(x, y, c, val);
                } else {
                    float prev_channel_sum = filtered.get_pixel(x, y, 0);
                    filtered.set_pixel(x, y, 0, prev_channel_sum+val);
                }
            }
        }
    }
    return filtered;
}

Image make_gaussian_filter(float sigma, bool normalize)
{
    int size = std::ceil(6 * sigma);
    if (size % 2 == 0)
        size++;
    int center = size / 2;
    Image filter(size, size, 1);
    double sum = 0;
    for (int x = -size/2; x <= size/2; x++) {
        for (int y = -size/2; y <= size/2; y++) {
            float val = std::exp(-(x*x + y*y) / (2*sigma*sigma));
            filter.set_pixel(center+x, center+y, 0, val);
            sum += val;
        }
    }
    if (normalize) {
        for (int i = 0; i < size*size; i++)
            filter.data[i] /= sum;
    }
    return filter;
}

// separable 2D gaussian blur for 1 channel image
Image gaussian_blur(const Image& img, float sigma)
{
    //assert(img.channels == 1);
    int size = std::ceil(6 * sigma);
    if (size % 2 == 0)
        size++;
    int center = size / 2;
    Image kernel(size, 1, 1);
    float sum = 0;
    for (int k = -size/2; k <= size/2; k++) {
        float val = std::exp(-(k*k) / (2*sigma*sigma));
        kernel.set_pixel(center+k, 0, 0, val);
        sum += val;
    }
    for (int k = 0; k < size; k++)
            kernel.data[k] /= sum;
    Image tmp(img.width, img.height, 1);
    Image filtered(img.width, img.height, 1);

    // vertial
    for (int x = 0; x < img.width; x++) {
        for (int y = 0; y < img.height; y++) {
            float sum = 0;
            for (int k = 0; k < size; k++) {
                int dy = -center + k;
                sum += img.get_pixel(x, y+dy, 0) * kernel.data[k];
            }
            tmp.set_pixel(x, y, 0, sum);
        }
    }
    // horizontal
    for (int x = 0; x < img.width; x++) {
        for (int y = 0; y < img.height; y++) {
            float sum = 0;
            for (int k = 0; k < size; k++) {
                int dx = -center + k;
                sum += tmp.get_pixel(x+dx, y, 0) * kernel.data[k];
            }
            filtered.set_pixel(x, y, 0, sum);
        }
    }
    return filtered;
}