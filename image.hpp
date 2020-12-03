#include <string>
#ifndef IMAGE_H
#define IMAGE_H

struct Image {
    Image(std::string file_path);
    Image(int w, int h, int c);
    ~Image();
    int width;
    int height;
    int channels;
    float *data;
    bool load();
    bool save(std::string file_path);
    void set_pixel(int x, int y, int c, float val);
    float get_pixel(int x, int y, int c);
    void clamp();
};

#endif
