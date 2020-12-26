#include <cmath>
#include <iostream>
#include <vector>
#include "sift.hpp"
#include "image.hpp"

#include <unistd.h>

namespace sift {

ScaleSpacePyramid generate_scale_space_pyramid(const Image& img, float base_sigma)
{
    // assume initial sigma is 1.0 (after resizing) and smooth
    // the image with sigma_diff to reach requried smoothing
    Image base_img = img.resize(img.width*2, img.height*2, Interpolation::BILINEAR);
    float sigma_diff = std::sqrt(base_sigma*base_sigma - 1.0f);
    base_img = convolve(base_img, make_gaussian_filter(sigma_diff), false);

    int num_octaves = 4, scales_per_octave = 3;
    int imgs_per_octave = scales_per_octave + 3;

    // determine all sigma values for all
    // and generate gaussian kernels for all sigmas
    float k = std::pow(2, 1.0/scales_per_octave);
    std::vector<float> sigma_vals {base_sigma};
    for (int i = 1; i < imgs_per_octave; i++) {
        float sigma_prev = base_sigma * std::pow(k, i-1);
        float sigma_total = k * sigma_prev;
        sigma_vals.push_back(std::sqrt(sigma_total*sigma_total - sigma_prev*sigma_prev));
    }
    std::vector<Image> gaussian_kernels;
    for (int i = 1; i < sigma_vals.size(); i++) {
        gaussian_kernels.push_back(make_gaussian_filter(sigma_vals[i]));
    }

    // create a scale space pyramid of gaussian images
    // images in each octave are 2x smaller than in the previous
    ScaleSpacePyramid pyramid = {
        .num_octaves = num_octaves,
        .imgs_per_octave = imgs_per_octave,
        .octaves = std::vector<std::vector<Image>>(num_octaves)
    };
    for (int i = 0; i < num_octaves; i++) {
        pyramid.octaves[i].reserve(imgs_per_octave);
        pyramid.octaves[i].push_back(std::move(base_img));
        for (const Image& kernel : gaussian_kernels) {
            const Image& prev_img = pyramid.octaves[i].back();
            pyramid.octaves[i].push_back(convolve(prev_img, kernel, false));
        }
        // prepare base image for next octave
        base_img = pyramid.octaves[i][imgs_per_octave-3]; //TODO reference, avoid copy
        base_img = base_img.resize(base_img.width/2, base_img.height/2, Interpolation::NEAREST);
    }
    return pyramid;
}

// generate pyramid of difference of gaussians (DoG) images
DoGPyramid generate_dog_pyramid(const ScaleSpacePyramid& img_pyramid)
{
    DoGPyramid dog_pyramid = {
        .num_octaves = img_pyramid.num_octaves,
        .imgs_per_octave = img_pyramid.imgs_per_octave - 1,
        .octaves = std::vector<std::vector<Image>>(img_pyramid.num_octaves)
    };
    for (int i = 0; i < dog_pyramid.num_octaves; i++) {
        dog_pyramid.octaves[i].reserve(dog_pyramid.imgs_per_octave);
        for (int j = 1; j < img_pyramid.imgs_per_octave; j++) {
            Image diff = img_pyramid.octaves[i][j];
            for (int pix_idx = 0; pix_idx < diff.size; pix_idx++) {
                diff.data[pix_idx] -= img_pyramid.octaves[i][j-1].data[pix_idx];
            }
            dog_pyramid.octaves[i].push_back(diff);
        }
    }
    return dog_pyramid;
}

bool point_is_extremum(const Image& img, const Image& prev, const Image& next, int x, int y)
{
    bool is_min = true, is_max = true;
    float val = img.get_pixel(x, y, 0), neighbor;

    for (int dx : {-1,0,1}) {
        for (int dy : {-1,0,1}) {
            neighbor = prev.get_pixel(x+dx, y+dy, 0);
            if (neighbor > val) is_max = false;
            if (neighbor < val) is_min = false;

            neighbor = next.get_pixel(x+dx, y+dy, 0);
            if (neighbor > val) is_max = false;
            if (neighbor < val) is_min = false;

            if (dx != 0 || dy != 0) {
                neighbor = img.get_pixel(x+dx, y+dy, 0);
                if (neighbor > val) is_max = false;
                if (neighbor < val) is_min = false;
            }
        }
    }
    return is_max || is_min;
}

std::vector<Keypoint> find_scalespace_extrema(const DoGPyramid& dog_pyramid)
{
    std::vector<Keypoint> extrema;
    for (int i = 0; i < dog_pyramid.num_octaves; i++) {
        for (int j = 1; j < dog_pyramid.imgs_per_octave-1; j++) {
            const Image& img = dog_pyramid.octaves[i][j];
            const Image& prev = dog_pyramid.octaves[i][j-1];
            const Image& next = dog_pyramid.octaves[i][j+1];
            for (int x = 1; x < img.width-1; x++) {
                for (int y = 1; y < img.height-1; y++) {
                    if (point_is_extremum(img, prev, next, x, y)) {
                        extrema.push_back({x, y, i}); //TODO: add sigma to keypoint struct
                    }
                }
            }
        }
    }
    std::cout << "num extrema: " << extrema.size() << "\n";
    return extrema;
}

} //namespace sift
