#include <iostream> 
#include <string>

#include "image.hpp"
#include "sift.hpp"

int main(int argc, char *argv[])
{
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(NULL);

    if (argc != 3) {
        std::cerr << "Usage: ./match_features a.jpg b.jpg (or .png)\n";
        return 0;
    }
    Image a(argv[1]), b(argv[2]);
    a = a.channels == 1 ? a : rgb_to_grayscale(a);
    b = b.channels == 1 ? b : rgb_to_grayscale(b);

    std::vector<sift::Keypoint> kps_a = sift::find_keypoints_and_descriptors(a);
    std::vector<sift::Keypoint> kps_b = sift::find_keypoints_and_descriptors(b);
    std::vector<std::pair<int, int>> matches = sift::find_keypoint_matches(kps_a, kps_b);
    Image result = sift::draw_matches(a, b, kps_a, kps_b, matches);
    result.save("result.jpg");
    
    std::cout << "Found " << matches.size() << " feature matches. Output image is saved as result.jpg\n";
    return 0;
}