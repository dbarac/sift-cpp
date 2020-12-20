#include <iostream> 
#include <string>

#include "image.hpp"
#include <cmath>
#include <string>
#include "sift.hpp"

int main(int argc, char *argv[])
{
    /*
    Image img = Image("plans2.png");
    std::cout << img.height << " " << img.width << " " <<  img.channels;
    Image gaussian = make_gaussian_filter(3);
    Image filtered = convolve(img, gaussian, true);
    filtered.save("filtered.jpg");
    */
    Image img("box.png");
    
    sift::ScaleSpacePyramid pyramid = sift::generate_scale_space_pyramid(img, 1.6);
    auto dog_pyramid = sift::generate_dog_pyramid(pyramid);
    int i  = 0;
    for (auto& img : dog_pyramid.octaves[2])
        img.save(std::to_string(++i)+".jpg");
	return 0;
}
