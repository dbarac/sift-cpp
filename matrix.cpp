#include <cassert>
#include <iostream>
#include "matrix.hpp"

namespace mat {

Matrix mul(const Matrix& a, const Matrix& b)
{
    assert(a.cols == b.rows);
    Matrix c(a.rows, b.cols);

    for (int i = 0; i < a.rows; i++) {
        for (int j = 0; j < b.cols; j++) {
            float val = 0.f;
            for (int k = 0; k < a.cols; k++) {
                val += a(i,k) * b(k,j);
            }
            c(i, j) = val;
        }
    }
    return c;
}

Matrix Matrix::transpose()
{
    Matrix tp(cols, rows);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            tp(j, i) = (*this)(i, j);
        }
    }
    return tp;
}

Matrix Matrix::invert()
{
    assert(rows == cols);

    // augment matrix with identity matrix
    Matrix aug(rows, cols*2);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            aug(i, j) = (*this)(i, j);
        }
        aug(i, cols+i) = 1;
    }

    // invert with Gauss-Jordan elimination
    for (int k = 0; k < cols; k++) {
        double abs_pivot_val = 0.;
        int pivot_row = -1;
        for (int i = k; i < rows; i++) {
            double val = std::abs(aug(i, k));
            if (val > abs_pivot_val) {
                abs_pivot_val = val;
                pivot_row = i;
            }
        }
        // if the matrix is singular return a 0x0 matrix
        if (pivot_row == -1) {
            std::cout << "matrix is singular\n";
            return Matrix(0,0);
        }
        //swap row k and pivot row
        for (int j = 0; j < aug.cols; j++) {
            std::swap(aug(k, j), aug(pivot_row, j));
        }
        //div row k by pivot_val (first non-zero value in the row)
        double pivot_val = aug(k, k);
        aug(k, k) = 1;
        for (int j = k+1; j < aug.cols; j++) {
            aug(k, j) /= pivot_val;
        }
        // sub row k from rest of rows
        for (int i = k+1; i < aug.rows; i++) {
            double val = aug(i, k);
            aug(i, k) = 0;
            for (int j = k+1; j < aug.cols; j++) {
                aug(i, j) -= val * aug(k, j);
            }
        }
    }

    // make sure all values above diagonal are 0
    for (int k = rows-1; k >= 0; k--) {
        for (int i = 0; i < k; i++) {
            double val = aug(i, k);
            aug(i, k) = 0;
            for (int j = k+1; j < aug.cols; j++) {
                aug(i, j) -= val * aug(k, j);
            }
        }
    }

    Matrix inv(rows, cols);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            inv(i, j) = aug(i, cols+j);
        }
    }
    return inv;
}

void print(const Matrix& a)
{
    for (int i = 0; i < a.rows; i++) {
        for (int j = 0; j < a.cols; j++) {
            std::cout << a(i, j) << " ";
        }
        std::cout << "\n";
    }
    std::cout << "\n";
}

} //namespace mat
