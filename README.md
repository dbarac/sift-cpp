# sift-cpp

## Introduction
This is a C++ implementation of [SIFT](https://en.wikipedia.org/wiki/Scale-invariant_feature_transform), a feature detection algorithm.

## Libraries used
[stb_image](https://github.com/nothings/stb) and stb_image_write for loading and saving images. (included in this repo)

## Usage example
Find keypoints, match features in two images and save the result:
```cpp
#include <vector>
#include "image.hpp"
#include "sift.hpp"

int main()
{
    Image img("./../imgs/book_rotated.jpg");
    Image img2("./../imgs/book_in_scene.jpg");
    img = rgb_to_grayscale(img);
    img2 = rgb_to_grayscale(img2);
    std::vector<sift::Keypoint> kps1 = sift::find_keypoints_and_descriptors(img);
    std::vector<sift::Keypoint> kps2 = sift::find_keypoints_and_descriptors(img2);
    std::vector<std::pair<int, int>> matches = sift::find_keypoint_matches(kps1, kps2);
    Image book_matches = sift::draw_matches(img, img2, kps1, kps2, matches);
    book_matches.save("book_matches.jpg");
    return 0;
}
```

Result:
![Matching result](./imgs/book_matches.jpg)

## Build and run the examples
### Build
```bash
$ mkdir build/ && cd build && cmake .. && make
```
The executables will be in sift-cpp/bin/.

### Run
Find image keypoints, draw them and save the result:
```bash
$ cd bin/ && ./find_keypoints ../imgs/book_rotated.jpg
```
Input images can be .jpg or .png. Result image is saved as result.jpg

![Keypoints result](./imgs/book_keypoints.jpg)

Find keypoints in two images and match them, draw matches and save the result:
```bash
$ cd bin/ && ./match_features ../imgs/book_rotated.jpg ../imgs/book_in_scene.jpg
```
Result image is saved as result.jpg

## Useful links

* [SIFT paper](https://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf)
* [Anatomy of the SIFT method](http://www.ipol.im/pub/art/2014/82/article.pdf)
* [Blog post about a Python SIFT implementation](https://medium.com/@russmislam/implementing-sift-in-python-a-complete-guide-part-1-306a99b50aa5)