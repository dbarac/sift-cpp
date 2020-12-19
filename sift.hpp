#include <vector>
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

ScaleSpacePyramid generate_scale_space_pyramid(const Image& img, float sigma);
DoGPyramid generate_dog_pyramid(const ScaleSpacePyramid& img_pyramid);

}
#endif
