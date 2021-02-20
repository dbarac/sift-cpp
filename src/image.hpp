#ifndef IMAGE_H
#define IMAGE_H
#include <string>

enum Interpolation {BILINEAR, NEAREST};

struct Image {
    explicit Image(std::string file_path);
    Image(int w, int h, int c);
    Image();
    ~Image();
    Image(const Image& other);
    Image& operator=(const Image& other);
    Image(Image&& other);
    Image& operator=(Image&& other);
    int width;
    int height;
    int channels;
    int size;
    float *data;
    bool save(std::string file_path);
    void set_pixel(int x, int y, int c, float val);
    float get_pixel(int x, int y, int c) const;
    void clamp();
    Image resize(int new_w, int new_h, Interpolation method = BILINEAR) const;
};


float bilinear_interpolate(const Image& img, float x, float y, int c);
float nn_interpolate(const Image& img, float x, float y, int c);

Image rgb_to_grayscale(const Image& img);
Image grayscale_to_rgb(const Image& img);

Image gaussian_blur(const Image& img, float sigma);

void draw_point(Image& img, int x, int y, int size=3);
void draw_line(Image& img, int x1, int y1, int x2, int y2);


#endif
