#include <iostream> 
#include <string>

#include "image.hpp"
#include <cmath>
#include <string>
#include <vector>
#include <unistd.h>
#include "sift.hpp"
#include <chrono>


int main(int argc, char *argv[])
{
    //std::ios_base::sync_with_stdio(false);
    //std::cin.tie(NULL);
    /*
    Image img = Image("plans2.png");
    std::cout << img.height << " " << img.width << " " <<  img.channels;
    Image gaussian = make_gaussian_filter(3);
    Image filtered = convolve(img, gaussian, true);
    filtered.save("filtered.jpg");
    */
    Image img("imgs/box.png");
    /*Image proc = img.resize(img.width*2, img.height*2);
    float sigmas[] = {1.248, 1.22627, 1.545, 1.94659, 2.45225, 3.09};
    for (float sigma : sigmas) {
        proc = convolve(proc, make_gaussian_filter(sigma), true);
        proc.save("test.jpg");
        std::cout << sigma << "\n";
        //sleep(1);
    }*/
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    sift::ScaleSpacePyramid pyramid = sift::generate_scale_space_pyramid(img, 1.6);
    auto dog_pyramid = sift::generate_dog_pyramid(pyramid);

    auto keypoints = find_scalespace_extrema(dog_pyramid);
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::cout << "Time difference (sec) = "
              <<  (std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()) /1000000.0  <<std::endl;
	return 0;
}
