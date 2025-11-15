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

/* ------------------ MATRIX ------------------ */

Matrix* createMatrix(size_t rows, size_t cols) {
    Matrix* M = malloc(sizeof(Matrix));
    if (M == NULL) {
        return NULL;
    }
    M->view.data = calloc(rows * cols, sizeof(double));
    if (M->view.data == NULL) {
        free(M);
        return NULL;
    }
    M->view.n = rows;
    M->view.features = cols;
    return M;
}

void destroyMatrix(Matrix* M) {
    if (M == NULL) {
        return;
    }
    free(M->view.data);
    free(M);
}

Matrix* onesMatrix(size_t rows, size_t cols) {
    Matrix* M = createMatrix(rows, cols);
    for (size_t i = 0; i < rows * cols; i++) {
        M->view.data[i] = 1;
    }
    return M;
}

size_t indexMatrix(MatrixView* Mv, size_t i, size_t j) {
    assert(i < Mv->n && j < Mv->features);
    return i * Mv->features + j;
}

double matrixGet(MatrixView* Mv, size_t i, size_t j) {
    return Mv->data[indexMatrix(Mv, i, j)];
}

void matrixSet(MatrixView* Mv, size_t i, size_t j, double value) {
    Mv->data[indexMatrix(Mv, i, j)] = value;
}

void copyMatrixRow(size_t rowIdx, MatrixView* Mv, double c[]) {
    assert(rowIdx < Mv->n);
    for (size_t i = 0; i < Mv->features; i++) {
        c[i] = matrixGet(Mv, rowIdx, i);
    }
}

void copyMatrixCol(size_t colIdx, MatrixView* Mv, double c[]) {
    assert(colIdx < Mv->features);
    for (size_t i = 0; i < Mv->n; i++) {
        c[i] = matrixGet(Mv, i, colIdx);
    }
}

void transpose(MatrixView* M, MatrixView* Mt) {
    assert(M->n == Mt->features && M->features == Mt->n);
    for (size_t i = 0; i < M->n; i++) {
        for (size_t j = 0; j < M->features; j++) {
            matrixSet(Mt, j, i, matrixGet(M, i, j));
        }
    }
}

void matMul(MatrixView* A, MatrixView* B, MatrixView* C) {
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

void printMatrix(MatrixView* Mv) {
    for (size_t i = 0; i < Mv->n; i++) {
        printf("|%.2f", matrixGet(Mv, i, 0));
        for (size_t j = 1; j < Mv->features; j++) {
            printf(", %.2f", matrixGet(Mv, i, j));
        }
        printf("|\n");
    }
}

void getMatrixView(size_t k, Tensor* T, MatrixView* M) {
    if (T == NULL || M == NULL) {
        return;
    }
    assert(k < T->batch);

    /* point to the beginning of the k-th data block, running from: */
    /* [k * T->n * T->features, k * T->n * T->features + T->n * T->features] */
    M->data = &T->data[k * T->n * T->features];
    M->n = T->n;
    M->features = T->features;
}

void batchedMatMul(Tensor* X, MatrixView* W, Tensor* Y) {
    /* Modern C, Takeaway 5.5 #1: All variables should be initialized. */
    MatrixView kth_input = {};
    MatrixView kth_output = {};

    for (size_t k = 0; k < X->batch; k++) {
        getMatrixView(k, X, &kth_input);
        getMatrixView(k, Y, &kth_output);

        matMul(&kth_input, W, &kth_output);
    }
}

void printTensor(Tensor* T) {
    MatrixView Kth = {};

    for (size_t k = 0; k < T->batch; k++) {
        getMatrixView(k, T, &Kth);
        printMatrix(&Kth);
        printf("\n");
    }
}
