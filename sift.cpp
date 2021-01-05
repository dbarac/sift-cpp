#define _USE_MATH_DEFINES
#include <cmath>
#include <iostream>
#include <vector>
#include <algorithm>
#include <unistd.h>

#include "sift.hpp"
#include "image.hpp"
#include "matrix.hpp"


namespace sift {

ScaleSpacePyramid generate_scale_space_pyramid(const Image& img, float base_sigma)
{
    // assume initial sigma is 1.0 (after resizing) and smooth
    // the image with sigma_diff to reach requried base_sigma
    Image base_img = img.resize(img.width*2, img.height*2, Interpolation::BILINEAR);
    float sigma_diff = std::sqrt(base_sigma*base_sigma - 1.0f);
    base_img = convolve(base_img, make_gaussian_filter(sigma_diff), false);

    int num_octaves = 4, scales_per_octave = 3;
    int imgs_per_octave = scales_per_octave + 3;

    // determine all sigma values for all kernels
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

bool point_is_extremum(const std::vector<Image>& octave, int scale, int x, int y)
{
    const Image& img = octave[scale];
    const Image& prev = octave[scale-1];
    const Image& next = octave[scale+1];

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

mat::Matrix quadratic_fit(Keypoint& kp, const std::vector<Image>& octave, int scale)
{
    const Image& img = octave[scale];
    const Image& prev = octave[scale-1];
    const Image& next = octave[scale+1];

    float g11, g12, g13;
    float h11, h12, h13, h21, h22, h23, h31, h32, h33;
    float offset_s, offset_x, offset_y;
    int x = kp.i, y = kp.j;

    g11 = (next.get_pixel(x, y, 0) - prev.get_pixel(x, y, 0)) / 2;
    g12 = (img.get_pixel(x+1, y, 0) - img.get_pixel(x-1, y, 0)) / 2;
    g13 = (img.get_pixel(x, y+1, 0) - img.get_pixel(x, y-1, 0)) / 2;

    mat::Matrix grad(3, 1);
    grad.data = {g11, g12, g13};

    h11 = next.get_pixel(x, y, 0) + prev.get_pixel(x, y, 0) - 2*img.get_pixel(x, y, 0);
    h22 = img.get_pixel(x+1, y, 0) + img.get_pixel(x-1, y, 0) - 2*img.get_pixel(x, y, 0);
    h33 = img.get_pixel(x, y+1, 0) + img.get_pixel(x, y-1, 0) - 2*img.get_pixel(x, y, 0);
    h12 = (next.get_pixel(x+1, y, 0) - next.get_pixel(x-1, y, 0)
          -prev.get_pixel(x+1, y, 0) + prev.get_pixel(x-1, y, 0)) / 4;
    h13 = (next.get_pixel(x, y+1, 0) - next.get_pixel(x, y-1, 0)
          -prev.get_pixel(x, y+1, 0) + prev.get_pixel(x, y-1, 0)) / 4;
    h23 = (img.get_pixel(x+1, y+1, 0) - img.get_pixel(x+1, y-1, 0)
          -img.get_pixel(x-1, y+1, 0) + img.get_pixel(x-1, y-1, 0)) / 4;
    
    mat::Matrix hessian(3,3);
    hessian.data = {
        h11, h12, h13,
        h12, h22, h23,
        h13, h23, h33
    };
    mat::Matrix hessian_inv = hessian.invert();
    if (hessian_inv.rows == 0) {
        std::cout << "could not invert hessian, keypoint should be discarded\n";    
        return hessian_inv; 
    }
    for (auto& val : hessian_inv.data) val = -val;
    mat::Matrix offsets = mat::mul(hessian_inv, grad);

    mat::Matrix grad_tp = grad.transpose();
    for (auto& val : grad_tp.data) val *= 1/2;
    float interpolated_extrema_val = img.get_pixel(x, y, 0) + mat::mul(grad_tp, offsets)(0,0);
    kp.extremum_val = interpolated_extrema_val;
    return offsets;
}

bool point_is_on_edge(const Keypoint& kp, const std::vector<Image>& octave, float edge_thresh)
{
    const Image& img = octave[kp.scale];
    float h11, h12, h22;
    int x = kp.i, y = kp.j;
    h11 = img.get_pixel(x+1, y, 0) + img.get_pixel(x-1, y, 0) - 2*img.get_pixel(x, y, 0);
    h22 = img.get_pixel(x, y+1, 0) + img.get_pixel(x, y-1, 0) - 2*img.get_pixel(x, y, 0);
    h12 = (img.get_pixel(x+1, y+1, 0) - img.get_pixel(x+1, y-1, 0)
          -img.get_pixel(x-1, y+1, 0) + img.get_pixel(x-1, y-1, 0)) / 4;

    float det_hessian = h11*h22 - h12*h12;
    float tr_hessian = h11 + h22;
    float edgeness = tr_hessian*tr_hessian / det_hessian;

    if (edgeness > std::pow(edge_thresh+1, 2)/edge_thresh)
        return true;
    else
        return false;
}

void find_absolute_keypoint_coords(Keypoint& kp, mat::Matrix offsets,
                                   int i, float sigma_min=0.8,
                                   float min_pix_dist=0.5, int n_spo=3)
{  
    kp.sigma = std::pow(2, i) * sigma_min * std::pow(2, (offsets(0, 0)+kp.scale)/n_spo);
    kp.x = min_pix_dist * std::pow(2, i) * (offsets(1, 0)+kp.i);
    kp.y = min_pix_dist * std::pow(2, i) * (offsets(2, 0)+kp.j);
}

bool refine_or_discard_keypoint(Keypoint& kp, const std::vector<Image>& octave,
                                int octave_idx, float contrast_thresh)
{
    int k = 0;
    bool kp_is_valid = false; 
    while (k++ < MAX_REFINEMENT_ITERS) {
        mat::Matrix offsets = quadratic_fit(kp, octave, kp.scale);
        if (offsets.rows == 0) {
            break; // could not fit quadratic function, discard keypoint
        }
        float max_offset = std::max({std::abs(offsets(0, 0)),
                                     std::abs(offsets(1, 0)),
                                     std::abs(offsets(2, 0))});
        // find nearest discrete coordinates
        kp.scale += std::round(offsets(1, 0));
        kp.i += std::round(offsets(1, 0));
        kp.j += std::round(offsets(2, 0));
        if (kp.scale >= octave.size()-1 || kp.scale < 1)
            break;

        bool valid_contrast = std::abs(kp.extremum_val) > contrast_thresh;
        if (max_offset < 0.6 && valid_contrast && !point_is_on_edge(kp, octave, 10.f)) {
            find_absolute_keypoint_coords(kp, offsets, octave_idx);
            kp_is_valid = true;
            break;
        }
    }
    return kp_is_valid;
}

std::vector<Keypoint> find_scalespace_extrema(const DoGPyramid& dog_pyramid, float contrast_thresh)
{
    std::vector<Keypoint> extrema;
    for (int i = 0; i < dog_pyramid.num_octaves; i++) {
        const std::vector<Image>& octave = dog_pyramid.octaves[i];
        for (int j = 1; j < dog_pyramid.imgs_per_octave-1; j++) {
            const Image& img = octave[j];
            for (int x = 1; x < img.width-1; x++) {
                for (int y = 1; y < img.height-1; y++) {
                    if (std::abs(img.get_pixel(x, y, 0)) < 0.8*contrast_thresh) {
                        continue;
                    }
                    if (point_is_extremum(octave, j, x, y)) {
                        Keypoint kp = {x, y, i, j, -1, -1, -1, -1};
                        bool kp_is_valid = refine_or_discard_keypoint(kp, octave, i, contrast_thresh);
                        if (kp_is_valid) {
                            extrema.push_back(kp);
                        }
                    }
                }
            }
        }
    }
    std::cout << "num extrema: " << extrema.size() << "\n";
    return extrema;
}

// calculate x derivatives for all images in the input pyramid
ScaleSpacePyramid generate_gx_pyramid(const ScaleSpacePyramid& pyramid)
{
    ScaleSpacePyramid gx_pyramid = {
        .num_octaves = pyramid.num_octaves,
        .imgs_per_octave = pyramid.imgs_per_octave,
        .octaves = std::vector<std::vector<Image>>(pyramid.num_octaves)
    };
    for (int i = 0; i < pyramid.num_octaves; i++) {
        gx_pyramid.octaves[i].reserve(gx_pyramid.imgs_per_octave);
        int width = pyramid.octaves[i][0].width;
        int height = pyramid.octaves[i][0].height;
        for (int j = 0; j < pyramid.imgs_per_octave; j++) {
            Image img(width, height, 1);
            for (int x = 1; x < img.width-1; x++) {
                for (int y = 1; y < img.height-1; y++) {
                    float val = (pyramid.octaves[i][j].get_pixel(x+1, y, 0)
                                -pyramid.octaves[i][j].get_pixel(x-1, y, 0)) / 2;
                    img.set_pixel(x, y, 0, val);
                }
            }
            gx_pyramid.octaves[i].push_back(img);
        }
    }
    return gx_pyramid;
}

// calculate y derivatives for all images in the input pyramid
ScaleSpacePyramid generate_gy_pyramid(const ScaleSpacePyramid& pyramid)
{
    ScaleSpacePyramid gy_pyramid = {
        .num_octaves = pyramid.num_octaves,
        .imgs_per_octave = pyramid.imgs_per_octave,
        .octaves = std::vector<std::vector<Image>>(pyramid.num_octaves)
    };
    for (int i = 0; i < pyramid.num_octaves; i++) {
        gy_pyramid.octaves[i].reserve(gy_pyramid.imgs_per_octave);
        int width = pyramid.octaves[i][0].width;
        int height = pyramid.octaves[i][0].height;
        for (int j = 0; j < pyramid.imgs_per_octave; j++) {
            Image img(width, height, 1);
            for (int x = 1; x < img.width-1; x++) {
                for (int y = 1; y < img.height-1; y++) {
                    float val = (pyramid.octaves[i][j].get_pixel(x, y+1, 0)
                                -pyramid.octaves[i][j].get_pixel(x, y-1, 0)) / 2;
                    img.set_pixel(x, y, 0, val);
                }
            }
            gy_pyramid.octaves[i].push_back(img);
        }
    }
    return gy_pyramid;
}

std::vector<float> find_keypoint_orientations(Keypoint& kp, 
                                              const ScaleSpacePyramid& gx_pyramid,
                                              const ScaleSpacePyramid& gy_pyramid)
{
    const int n_bins = 36;
    const float min_pix_dist = 0.5;
    const float lambda_ori = 1.5;
    const float pix_dist = min_pix_dist * std::pow(2, kp.octave);
    
    //derivative images
    const Image& img_gx = gx_pyramid.octaves[kp.octave][kp.scale];
    const Image& img_gy = gy_pyramid.octaves[kp.octave][kp.scale];
    //int x = kp.x / pix_dist;
    //int y = kp.y / pix_dist;

    // discard kp if too close to image borders 
    float min_dist_from_border = std::min({kp.x, kp.y, pix_dist*img_gx.width-kp.x, pix_dist*img_gx.height-kp.y});
    if (min_dist_from_border <= 3*lambda_ori*kp.sigma) {
        //std::cout << "too close to border\n";
        return std::vector<float>();
    }
    //find patch size w.r.t. the keypoint octave image size
    int patch_size = std::round(6 * lambda_ori * kp.sigma / pix_dist);
    if (patch_size % 2 == 0)
        patch_size++;

    // accumulate gradients in orientation histogram
    float hist[n_bins] = {};
    int bin;
    float gx, gy, grad_norm, weight, patch_sigma, theta;
    
    int x_start = std::round((kp.x - 3*lambda_ori*kp.sigma)/pix_dist);
    int x_end = std::round((kp.x + 3*lambda_ori*kp.sigma)/pix_dist);
    int y_start = std::round((kp.y - 3*lambda_ori*kp.sigma)/pix_dist);
    int y_end = std::round((kp.y + 3*lambda_ori*kp.sigma)/pix_dist);

    //std::cout << "patch: " << patch_size << " " << x_end-x_start+1 << "\n";
    for (int x = x_start; x <= x_end; x++) {
        for (int y = y_start; y <= y_end; y++) {
            gx = img_gx.get_pixel(x, y, 0);
            gy = img_gy.get_pixel(x, y, 0);
            grad_norm = std::sqrt(gx*gx + gy*gy);
            patch_sigma = lambda_ori * kp.sigma;
            weight = std::exp(-(std::pow(x*pix_dist-kp.x, 2)+std::pow(y*pix_dist-kp.y, 2))
                              /(2*patch_sigma*patch_sigma));
            theta = std::fmod(std::atan2(gy, gx)+2*M_PI, 2*M_PI);
            bin = std::round(n_bins/(2*M_PI) * theta);
            hist[bin] += weight * grad_norm;
        }
    }

    // smooth histogram
    float tmp_hist[n_bins];
    for (int i = 0; i < 6; i++) {
        for (int j = 0; j < n_bins; j++) {
            int prev_idx = (j-1+n_bins)%n_bins;
            int next_idx = (j+1)%n_bins;
            tmp_hist[j] = (hist[prev_idx] + hist[j] + hist[next_idx]) / 3;
        }
        for (int j = 0; j < n_bins; j++) {
            hist[j] = tmp_hist[j];
        }
    }

    // extract reference orientations
    float ori_thresh = 0.8, ori_max = 0;
    int max_idx = -1;
    std::vector<float> orientations;
    for (int j = 0; j < n_bins; j++) {
        if (hist[j] > ori_max) {
            max_idx = j;
            ori_max = hist[j];
        }
    }
    for (int j = 0; j < n_bins; j++) {
        if (hist[j] > ori_thresh * ori_max) {
            float prev = hist[(j-1+n_bins)%n_bins], next = hist[(j+1)%n_bins];
            if (prev > hist[j] || next > hist[j])
                continue;
            float theta = 2*M_PI*(j+1)/n_bins + M_PI/n_bins*(prev-next)/(prev-2*hist[j]+next); // j+1?
            orientations.push_back(theta);
        }
    }
    return orientations;
}

void compute_keypoint_descriptor(Keypoint& kp, float orientation,
                                 const ScaleSpacePyramid& gx_pyramid,
                                 const ScaleSpacePyramid& gy_pyramid)
{
    const float min_pix_dist = 0.5;
    const float lambda_desc = 6;
    const float pix_dist = min_pix_dist * std::pow(2, kp.octave);
    const int n_hist = 4, n_ori = 8;
    
    //derivative images
    const Image& img_gx = gx_pyramid.octaves[kp.octave][kp.scale];
    const Image& img_gy = gy_pyramid.octaves[kp.octave][kp.scale];

    // discard kp if too close to image borders 
    float min_dist_from_border = std::min({kp.x, kp.y, pix_dist*img_gx.width-kp.x, pix_dist*img_gx.height-kp.y});
    if (min_dist_from_border <= std::sqrt(2)*lambda_desc*kp.sigma) {
        //std::cout << "too close to border\n";
        return;
    }

    
}

} //namespace sift
