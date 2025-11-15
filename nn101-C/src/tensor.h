#ifndef TENSOR_H
#define TENSOR_H

#include <stddef.h>

typedef struct Tensor Tensor;
struct Tensor {
    double* data;
    size_t batch;
    size_t n;
    size_t features;
};

/* Non-owner structure: do not call free on `.data`. Used to index into larger
 * owner structures. */
typedef struct MatrixView MatrixView;
struct MatrixView {
    double* data;
    size_t n;
    size_t features;
};

/* Owner structure(unused atm): must call free on `.data`.typedef struct Matrix
 * Matrix; */
typedef struct Matrix Matrix;
struct Matrix {
    MatrixView view;
};

Tensor* createTensor(size_t batch_dim, size_t dim0, size_t dim1);
void destroyTensor(Tensor* T);

size_t indexTensor(Tensor* T, size_t k, size_t i, size_t j);
void printTensor(Tensor* T);
double tensorGet(Tensor* T, size_t k, size_t i, size_t j);
/* void tensorSet(Tensor* T, size_t k, size_t i, size_t j, double value); */

size_t indexMatrix(MatrixView* Mv, size_t i, size_t j);
double matrixGet(MatrixView* Mv, size_t i, size_t j);
void matrixSet(MatrixView* Mv, size_t i, size_t j, double value);
void printMatrix(MatrixView* Mv);
void transpose(MatrixView* M, MatrixView* Mt);
void matMul(MatrixView* A, MatrixView* B, MatrixView* C);
void copyMatrixRow(size_t rowIdx, MatrixView* Mv, double c[]);
void copyMatrixCol(size_t colIdx, MatrixView* Mv, double c[]);

Matrix* createMatrix(size_t rows, size_t cols);
void destroyMatrix(Matrix* M);
Matrix* onesMatrix(size_t rows, size_t cols);

void getMatrixView(size_t k, Tensor* T, MatrixView* M);
void batchedMatMul(Tensor* X, MatrixView* W, Tensor* Y);

#endif  // TENSOR_H
