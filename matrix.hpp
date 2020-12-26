#ifndef MATRIX_H
#define MATRIX_H

#include <vector>
#include <initializer_list>

namespace mat {

struct Matrix {
    int rows;
    int cols;
    std::vector<float> data;
    Matrix(int m, int n): rows {m}, cols {n}, data {std::vector<float>(m*n)} {}
    //Matrix(int m, int n, std::initializer_list<float> elements): rows {m}, cols {n}, data {elements} {}
    float& operator() (int i, int j) {return data[i*cols+j];}
    float operator() (int i, int j) const {return data[i*cols+j];}
    Matrix transpose();
    Matrix invert();
};

Matrix mul(const Matrix& a, const Matrix& b);
void print(const Matrix& a);

} //namespace mat
#endif
