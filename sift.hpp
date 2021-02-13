#include <vector>
#include <array>
#include "image.hpp"
#ifndef SIFT_H
#define SIFT_H

namespace sift {

//scale space gaussian pyramid
struct ScaleSpacePyramid {
    int num_octaves;
    int imgs_per_octave;
    std::vector<std::vector<Image>> octaves; 
};

//pyramid of difference of gaussian images
struct DoGPyramid {
    int num_octaves;
    int imgs_per_octave;
    std::vector<std::vector<Image>> octaves;
};

struct Keypoint {
    // discrete coordinates
    int i;
    int j;
    int octave;
    int scale; //index of gaussian image inside the octave

    // continuous coordinates (interpolated)
    float x;
    float y;
    float sigma;
    float extremum_val; //value of interpolated DoG extremum
    
    std::array<int, 128> descriptor;
};

const int MAX_REFINEMENT_ITERS = 5;

ScaleSpacePyramid generate_scale_space_pyramid(const Image& img, float sigma);
DoGPyramid generate_dog_pyramid(const ScaleSpacePyramid& img_pyramid);
std::vector<Keypoint> find_scalespace_extrema(const DoGPyramid& dog_pyramid, float contrast_thresh=0.015);

ScaleSpacePyramid generate_gx_pyramid(const ScaleSpacePyramid& pyramid);
ScaleSpacePyramid generate_gy_pyramid(const ScaleSpacePyramid& pyramid);
std::vector<float> find_keypoint_orientations(Keypoint& kp, const ScaleSpacePyramid& gx_pyramid, const ScaleSpacePyramid& gy_pyramid);
std::array<int, 128> compute_keypoint_descriptor(Keypoint& kp, float theta, const ScaleSpacePyramid& gx_pyramid, const ScaleSpacePyramid& gy_pyramid);

std::vector<Keypoint> find_keypoints_and_descriptors(const Image& img);
std::vector<std::pair<int, int>> find_keypoint_matches(std::vector<Keypoint>& a, std::vector<Keypoint>& b);

Image draw_matches(const Image& a, const Image& b, std::vector<Keypoint> kps_a, std::vector<Keypoint> kps_b, std::vector<std::pair<int, int>> matches);

}
#endif
