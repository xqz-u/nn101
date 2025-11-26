#ifndef TENSOR_H
#define TENSOR_H

#include <gsl/gsl_rng.h>
#include <stdarg.h>
#include <stddef.h>

typedef struct Tensor Tensor;
struct Tensor {
    double* data;
    size_t batch;
    size_t n;
    size_t features;
    size_t flat_size;

    bool owns_data;

    void (*printShape)(Tensor* T);
};

Tensor* createTensor(size_t batch_dim, size_t dim0, size_t dim1);
void destroyTensor(Tensor* T);
void destroyTensors(size_t n, ...);
size_t indexTensor(Tensor* T, size_t k, size_t i, size_t j);
double tensorGet(Tensor* T, size_t k, size_t i, size_t j);
void printMatrix(Tensor* M);
void printTensor(Tensor* T);
/* void getMatrixAtIndex(size_t k, Tensor* T, Tensor* M); */
void flattenFirstDims(Tensor* T, Tensor* T_view);
void tensorFill(Tensor* T, double value);
Tensor* zerosLike(Tensor* T);
Tensor* onesLike(Tensor* T);
void randUniformTensor(Tensor* T, double center, gsl_rng* rng);

Tensor* createVector(size_t size);
Tensor* onesVector(size_t size);
double dotProduct(Tensor* a, Tensor* b);
void addTensorAndVector(Tensor* X, Tensor* b);

Tensor* createMatrix(size_t rows, size_t cols);
Tensor* onesMatrix(size_t rows, size_t cols);
size_t indexMatrix(Tensor* M, size_t i, size_t j);
double matrixGet(Tensor* M, size_t i, size_t j);
void matrixSet(Tensor* M, size_t i, size_t j, double value);
void copyMatrixRow(size_t rowIdx, Tensor* M, double c[]);
void copyMatrixCol(size_t colIdx, Tensor* M, double c[]);
void transpose(Tensor* M, Tensor* Mt);
void matMul(Tensor* A, Tensor* B, Tensor* C);
void batchMatMul(Tensor* X, Tensor* W, Tensor* Y);
void matrixSum(Tensor* M, size_t dim, Tensor* v);

void tensorAddScaled(Tensor* A, double scale, Tensor* B);
void ensureTensorCapacity(Tensor** source, Tensor* reference);
size_t argmax(Tensor* t);

#endif  // TENSOR_H
