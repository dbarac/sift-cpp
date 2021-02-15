#include <iostream> 
#include <string>

#include "image.hpp"
#include <cmath>
#include <string>
#include <vector>
#include <unistd.h>
#include "sift.hpp"
#include <chrono>
#include "matrix.hpp"

int main(int argc, char *argv[])
{
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(NULL);

    Image img("imgs/book_rotated.jpg");
    Image img2("imgs/book_in_scene.jpg");
    img = rgb_to_grayscale(img);
    img2 = rgb_to_grayscale(img2);

    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

    auto kps_a = sift::find_keypoints_and_descriptors(img);
    auto kps_b = sift::find_keypoints_and_descriptors(img2);
    auto matches = sift::find_keypoint_matches(kps_a, kps_b);
    std::cout << "matches: " << matches.size() << "\n";
    
    //Image blur1 = convolve(img, make_gaussian_filter(5), true);
    //Image blur2 = gaussian_blur(img, 5);
    //blur1.clamp();
    //Image blur1 = img;
    //blur1.save("blur1.jpg");
    //blur2.save("blur2.jpg");
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();


    std::cout << "Time difference (sec) = "
              <<  (std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()) /1000000.0  << "\n";
	
    Image rgb = grayscale_to_rgb(img);
    int out_of_bounds = 0;
    std::cout << "kps: " << kps_a.size() << "\n";
    for (auto& kp : kps_a) {
        draw_point(rgb, kp.x, kp.y);
    }
    rgb.save("box_keypoints.jpg");

    rgb = grayscale_to_rgb(img2);
    std::cout << "kps: " << kps_b.size() << "\n";
    for (auto& kp : kps_b) {
        draw_point(rgb, kp.x, kp.y);
    }
    rgb.save("box_scene_keypoints.jpg");

    Image box_matches = sift::draw_matches(img, img2, kps_a, kps_b, matches);
    box_matches.save("book_matches.jpg");
    return 0;
}
