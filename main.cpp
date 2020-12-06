#include <iostream> 
#include <string>

#include "image.hpp"

int main(int argc, char *argv[])
{
    Image img = Image("plans2.png");
    std::cout << img.height << " " << img.width << " " <<  img.channels;
    Image gaussian = make_gaussian_filter(3);
    Image filtered = convolve(img, gaussian, true);
    filtered.save("filtered.jpg");
	return 0;
}
