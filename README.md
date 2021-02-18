# sift-cpp

## Introduction
This is a C++ implementation of [SIFT](https://en.wikipedia.org/wiki/Scale-invariant_feature_transform), a feature detection algorithm.

Useful links for learning about SIFT:

## Libraries used
[stb_image](https://github.com/nothings/stb) and stb_image_write for loading and saving images. (included in this repo)

## Usage example
Find keypoints, match features in two images and save the result:
```cpp
#include "image.hpp"
#include "sift.hpp"

int main()
{
    Image img("./../imgs/book_rotated.jpg");
    Image img2("./../imgs/book_in_scene.jpg");
    img = rgb_to_grayscale(img);
    img2 = rgb_to_grayscale(img2);
    std::vector<sift::Keypoint> kps_a = sift::find_keypoints_and_descriptors(img);
    std::vector<sift::Keypoint> kps_b = sift::find_keypoints_and_descriptors(img2);
    std::vector<std::pair<int, int>> matches = sift::find_keypoint_matches(kps_a, kps_b);
    Image box_matches = sift::draw_matches(img, img2, kps_a, kps_b, matches);
    box_matches.save("./../imgs/book_matches.jpg");
    return 0;
}
```

Result:
![Matching result](./imgs/book_matches.jpg)

## Build and run
Build:
```bash
$ mkdir build/ && cd build && cmake .. && make
```
The executable will be in project_root/bin/.

Run:
```bash
$ cd bin/ && ./main
```
Result images will be saved in imgs/.