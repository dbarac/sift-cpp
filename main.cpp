#include <iostream> 
#include <string>

#include "image.hpp"

int main(int argc, char *argv[])
{
	//int width, height, bpp;
	//unsigned char* rgb = stbi_load("Rainier2.png", &width, &height, &bpp, 0);
	//std::cout << width << " " <<  height << " " << bpp;
	//stbi_image_free(rgb);
    Image img = Image("plans2.png");
    std::cout << img.height << " " << img.width << " " <<  img.channels;
    img.save("plans2-moje.jpg");
	return 0;
}
