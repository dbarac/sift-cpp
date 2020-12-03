#include "image.hpp"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

Image::Image(std::string file_path)
{
    int width, height, channels;
    unsigned char *img_data = stbi_load(file_path.c_str(), &width, &height, &channels, 0);
    this->width = width;
    this->height = height;
    if (channels == 4)
        channels = 3; //ignore alpha channel
    this->channels = channels;

    int size = width * height * channels;
    this->data = new float[size]; 
    for (int x = 0; x < width; x++) {
        for (int y = 0; y < height; y++) {
            for (int c = 0; c < channels; c++) {
                int src_idx = y*width*channels + x*channels + c;
                int dst_idx = c*height*width + y*width + x;
                this->data[dst_idx] = img_data[src_idx] / 255.;
            }
        }
    }
    stbi_image_free(img_data);
}

Image::Image(int w, int h, int c)
{
    width = w;
    height = h;
    channels = c;
    data = new float[w*h*c]();;
}

Image::~Image()
{
    delete[] this->data;
}

//save image as jpg file
bool Image::save(std::string file_path)
{
    unsigned char *out_data = new unsigned char[width*height*channels]; 
    for (int x = 0; x < width; x++) {
        for (int y = 0; y < height; y++) {
            for (int c = 0; c < channels; c++) {
                int dst_idx = y*width*channels + x*channels + c;
                int src_idx = c*height*width + y*width + x;
                out_data[dst_idx] = static_cast<unsigned char>(std::roundf(data[src_idx] * 255.));
            }
        }
    }
    bool success;
    success = stbi_write_jpg(file_path.c_str(), width, height, channels, out_data, 100);
    delete[] out_data;
    return true;
}

void Image::set_pixel(int x, int y, int c, float val)
{
    if (x > width || x < 0 || y > height || y < 0 || c > channels || c < 0)
        return;
    data[c*width*height + y*width + x] = val;
}

float Image::get_pixel(int x, int y, int c)
{
    if (x < 0)
        x = 0;
    if (x >= width)
        x = width - 1;
    if (y < 0)
        y = 0;
    if (y >= width)
        y = width - 1;
    return data[c*width*height + y*width + x];
}

void Image::clamp()
{
    int size = width * height * channels;
    for (int i = 0; i < size; i++) {
        float val = data[i];
        val = (val > 1.0) ? 1.0 : val;
        val = (val < 0.0) ? 0.0 : val;
        data[i] = val;
    }
}
