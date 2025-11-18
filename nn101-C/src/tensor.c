#include <assert.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

#include "tensor.h"

/* ------------------ TENSOR ------------------ */

Tensor* createTensor(size_t batch_dim, size_t dim0, size_t dim1) {
    Tensor* T = malloc(sizeof(Tensor));
    if (T == NULL) {
        return NULL;
    }
    T->data = calloc(batch_dim * dim0 * dim1, sizeof(double));
    if (T->data == NULL) {
        free(T);
        return NULL;
    };
    T->batch = batch_dim;
    T->n = dim0;
    T->features = dim1;
    T->flat_size = batch_dim * dim0 * dim1;
    return T;
}

void destroyTensor(Tensor* T) {
    if (T == NULL) {
        return;
    }
    free(T->data);
    free(T);
}

size_t indexTensor(Tensor* T, size_t k, size_t i, size_t j) {
    assert(T != NULL && k < T->batch && i < T->n && j < T->features);
    return k * T->n * T->features + i * T->features + j;
}

double tensorGet(Tensor* T, size_t k, size_t i, size_t j) {
    return T->data[indexTensor(T, k, i, j)];
}

/* void tensorSet(Tensor* T, size_t k, size_t i, size_t j, double value) { */
/*     T->data[indexTensor(T, k, i, j)] = value; */
/* } */

void printMatrix(Tensor* M) {
    printf("Shape: (%lu, %lu, %lu)\n", M->batch, M->n, M->features);
    for (size_t i = 0; i < M->n; i++) {
        printf("[");
        for (size_t j = 0; j < M->features; j++) {
            printf(" %10.3f", matrixGet(M, i, j));
        }
        printf(" ]\n");
    }
}

void printTensor(Tensor* T) {
    Tensor Kth = {};

    for (size_t k = 0; k < T->batch; k++) {
        getMatrixView(k, T, &Kth);
        printMatrix(&Kth);
        printf("\n");
    }
}

/* NOTE Do not call `destroyTensor(.)` on Tensors created by this function! */
void getMatrixView(size_t k, Tensor* T, Tensor* M) {
    if (T == NULL || M == NULL) {
        return;
    }
    assert(k < T->batch);

    /* point to the beginning of the k-th data block, running from: */
    /* [k * T->n * T->features, k * T->n * T->features + T->n * T->features] */
    M->data = &T->data[k * T->n * T->features];
    M->batch = 1;
    M->n = T->n;
    M->features = T->features;
}

Tensor* tensorLike(Tensor* T) {
    return createTensor(T->batch, T->n, T->features);
}

Tensor* onesTensor(size_t batch, size_t n, size_t features) {
    Tensor* T = createTensor(batch, n, features);
    for (size_t i = 0; i < T->flat_size; i++) {
        T->data[i] = 1.f;
    }
    return T;
}

/* ------------------ VECTOR ------------------ */

Tensor* createVector(size_t size) {
    return createTensor(1, 1, size);
}

double dotProduct(Tensor* a, Tensor* b) {
    assert(a->batch == 1 && b->batch == 1 && a->n == 1 && b->n == 1 &&
           a->features == b->features);
    double z = 0;
    for (size_t i = 0; i < a->features; i++) {
        z += a->data[i] * b->data[i];
    }
    return z;
}

void addTensorAndVector(Tensor* X, Tensor* b) {
    assert(X->features == b->features && b->batch == 1);
    for (size_t k = 0; k < X->batch; k++) {
        for (size_t i = 0; i < X->n; i++) {
            for (size_t j = 0; j < X->features; j++) {
                X->data[indexTensor(X, k, i, j)] += b->data[j];
            }
        }
    }
}

Tensor* onesVector(size_t size) {
    return onesTensor(1, 1, size);
}

/* ------------------ MATRIX ------------------ */

Tensor* createMatrix(size_t rows, size_t cols) {
    return createTensor(1, rows, cols);
}

Tensor* onesMatrix(size_t rows, size_t cols) {
    return onesTensor(1, rows, cols);
}

size_t indexMatrix(Tensor* M, size_t i, size_t j) {
    return indexTensor(M, 0, i, j);
}

double matrixGet(Tensor* M, size_t i, size_t j) {
    return M->data[indexMatrix(M, i, j)];
}

void matrixSet(Tensor* M, size_t i, size_t j, double value) {
    M->data[indexMatrix(M, i, j)] = value;
}

void copyMatrixRow(size_t rowIdx, Tensor* M, double c[]) {
    assert(M->batch == 1 && rowIdx < M->n);
    for (size_t i = 0; i < M->features; i++) {
        c[i] = matrixGet(M, rowIdx, i);
    }
}

void copyMatrixCol(size_t colIdx, Tensor* M, double c[]) {
    assert(M->batch == 1 && colIdx < M->features);
    for (size_t i = 0; i < M->n; i++) {
        c[i] = matrixGet(M, i, colIdx);
    }
}

void transpose(Tensor* M, Tensor* Mt) {
    assert(M->batch == 1 && Mt->batch == 1 && M->n == Mt->features &&
           M->features == Mt->n);
    for (size_t i = 0; i < M->n; i++) {
        for (size_t j = 0; j < M->features; j++) {
            matrixSet(Mt, j, i, matrixGet(M, i, j));
        }
    }
}

void matMul(Tensor* A, Tensor* B, Tensor* C) {
    assert(A->features == B->n);
    for (size_t c = 0; c < B->features; c++) {
        for (size_t i = 0; i < A->n; i++) {
            double C_ic = matrixGet(C, i, c);
            for (size_t j = 0; j < B->n; j++) {
                double A_ij = matrixGet(A, i, j);
                double B_jc = matrixGet(B, j, c);
                C_ic += A_ij * B_jc;
            }
            matrixSet(C, i, c, C_ic);
        }
    }
}

void batchedMatMul(Tensor* X, Tensor* W, Tensor* Y) {
    assert(W->batch == 1);
    /* Modern C, Takeaway 5.5 #1: All variables should be initialized. */
    Tensor kth_input = {};
    Tensor kth_output = {};

    for (size_t k = 0; k < X->batch; k++) {
        getMatrixView(k, X, &kth_input);
        getMatrixView(k, Y, &kth_output);

        matMul(&kth_input, W, &kth_output);
    }
}
