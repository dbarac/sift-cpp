#include <iostream> 
#include <string>
#include <chrono>

#include "image.hpp"
#include "sift_omp.hpp"

int main(int argc, char *argv[])
{
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(NULL);

    if (argc != 4) {
        std::cerr << "Usage: ./match_features a.jpg b.jpg out.jpg (or .png)\n";
        return 0;
    }
    Image a(argv[1]), b(argv[2]);
    a = a.channels == 1 ? a : rgb_to_grayscale(a);
    b = b.channels == 1 ? b : rgb_to_grayscale(b);

    auto tp1 = std::chrono::steady_clock::now();
    std::vector<siftomp::Keypoint> kps_a = siftomp::find_keypoints_and_descriptors(a);
    std::vector<siftomp::Keypoint> kps_b = siftomp::find_keypoints_and_descriptors(b);
    std::vector<std::pair<int, int>> matches = siftomp::find_keypoint_matches(kps_a, kps_b);
    auto tp2 = std::chrono::steady_clock::now();
    printf("Spend: %d microseconds on finding keypoints and matching them.\n", std::chrono::duration_cast<std::chrono::microseconds>(tp2 - tp1).count());
    Image result = siftomp::draw_matches(a, b, kps_a, kps_b, matches);
    result.save(argv[3]);
    
    std::cout << "Found " << matches.size() << " feature matches. Output image is saved as " << argv[3] << "\n";
    return 0;
}