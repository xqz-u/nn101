#include <assert.h>
#include <gsl/gsl_randist.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "tensor.h"

/* ------------------ TENSOR ------------------ */

void printTensorShape(Tensor* T) {
    if (T == nullptr) {
        return;
    }
    printf("Shape: (%lu, %lu, %lu) size %lu\n", T->batch, T->n, T->features,
           T->flat_size);
}

Tensor* createTensor(size_t batch_dim, size_t dim0, size_t dim1) {
    Tensor* T = malloc(sizeof(Tensor));
    if (T == nullptr) {
        return nullptr;
    }
    T->data = calloc(batch_dim * dim0 * dim1, sizeof(double));
    if (T->data == nullptr) {
        free(T);
        return nullptr;
    };
    T->batch = batch_dim;
    T->n = dim0;
    T->features = dim1;
    T->flat_size = batch_dim * dim0 * dim1;
    T->owns_data = true;
    T->printShape = &printTensorShape;
    return T;
}

void destroyTensor(Tensor* T) {
    if (T == nullptr) {
        return;
    }
    if (T->owns_data) {
        free(T->data);
    }
    free(T);
}

void destroyTensors(size_t n, ...) {
    va_list args;
    va_start(args, n);
    for (size_t i = 0; i < n; i++) {
        destroyTensor(va_arg(args, Tensor*));
    }
    va_end(args);
}

size_t indexTensor(Tensor* T, size_t k, size_t i, size_t j) {
    assert(T != nullptr && k < T->batch && i < T->n && j < T->features);
    return k * T->n * T->features + i * T->features + j;
}

double tensorGet(Tensor* T, size_t k, size_t i, size_t j) {
    return T->data[indexTensor(T, k, i, j)];
}

void getMatrixAtIndex(size_t k, Tensor* T, Tensor* M) {
    if (T == nullptr || M == nullptr) {
        return;
    }
    assert(k < T->batch);

    /* point to the beginning of the k-th data block running from */
    /* [k * T->n * T->features, k * T->n * T->features + T->n * T->features] */
    M->flat_size = T->n * T->features;
    M->batch = 1;
    M->n = T->n;
    M->features = T->features;
    M->data = &T->data[k * M->flat_size];
    M->owns_data = false;
    M->printShape = &printTensorShape;
}

/* Creates a view treating `T` as (rows, cols) with cols = last dimension, rows
 * = product of leading dimensions. */
void flattenFirstDims(Tensor* T, Tensor* T_view) {
    if (T == nullptr || T_view == nullptr) {
        return;
    }
    T_view->batch = 1;
    T_view->n = T->batch * T->n;
    T_view->features = T->features;
    T_view->data = T->data;
    T_view->flat_size = T->flat_size;
    T_view->owns_data = false;
    T_view->printShape = &printTensorShape;
}

void tensorFill(Tensor* T, double value) {
    assert(T != nullptr);
    if (value == 0.0) {
        memset(T->data, 0, T->flat_size * sizeof(double));
        return;
    }
    for (size_t i = 0; i < T->flat_size; i++) {
        T->data[i] = value;
    }
}

Tensor* onesTensor(size_t batch, size_t n, size_t features) {
    Tensor* T = createTensor(batch, n, features);
    tensorFill(T, 1);
    return T;
}

Tensor* zerosLike(Tensor* T) {
    return createTensor(T->batch, T->n, T->features);
}

Tensor* onesLike(Tensor* T) {
    return onesTensor(T->batch, T->n, T->features);
}

void randUniformTensor(Tensor* T, double center, gsl_rng* rng) {
    for (size_t i = 0; i < T->flat_size; i++) {
        T->data[i] = gsl_ran_flat(rng, -center, center);
    }
}

void printMatrixInternal(Tensor* M, bool shouldPrintShape) {
    if (M == nullptr) {
        return;
    }
    if (shouldPrintShape) {
        printTensorShape(M);
    }
    for (size_t i = 0; i < M->n; i++) {
        printf("[");
        for (size_t j = 0; j < M->features; j++) {
            printf(" %10.2f", matrixGet(M, i, j));
        }
        printf(" ]\n");
    }
}

void printMatrix(Tensor* M) {
    printMatrixInternal(M, true);
}

void printTensor(Tensor* T) {
    Tensor Kth = {};

    printTensorShape(T);
    for (size_t k = 0; k < T->batch; k++) {
        getMatrixAtIndex(k, T, &Kth);
        printMatrixInternal(&Kth, false);
        printf("\n");
    }
}

/* ------------------ VECTOR ------------------ */

Tensor* createVector(size_t size) {
    return createTensor(1, 1, size);
}

/* NOTE This works with tensors/matrices too, as long as they have the same
 * capacity. But it really is meaningful for vectors only. */
double dotProduct(Tensor* a, Tensor* b) {
    assert(a->flat_size == b->flat_size);

    double z = 0;
    for (size_t i = 0; i < a->flat_size; i++) {
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

void batchMatMul(Tensor* X, Tensor* W, Tensor* Y) {
    assert(W->batch == 1);

    /* Modern C takeaway 5.5 #1: All variables should be initialized. */
    Tensor X_view = {};
    Tensor Y_view = {};

    /* These tensor views are created only to 'fool' matMul, the result of
     * multiplication is the same, just the shapes differ:
     * (B, N, F) @ (F, C) == (B * N, F) @ (F, C). Using Y->data as underlying
     * storage for Y_view, the output still receives the correct expanded shape.
     * Generally speaking, every dimension except the last one is treated as as
     * n in the formula (n, m) x (m, k) -> (n, k). */
    flattenFirstDims(X, &X_view);
    flattenFirstDims(Y, &Y_view);

    matMul(&X_view, W, &Y_view);
}

/* NOTE Make sure that `v->data` is filed with 0s */
void matrixSum(Tensor* M, size_t dim, Tensor* v) {
    assert(M->batch == 1);
    assert(dim == 0 || dim == 1);
    assert(v != nullptr && v->batch == 1 && v->n == 1);

    if (dim == 0) {
        /* Sum over dimension 0 (row): sum each column */
        assert(v->features == M->features);
        for (size_t i = 0; i < M->n; i++) {
            for (size_t j = 0; j < M->features; j++) {
                v->data[j] += matrixGet(M, i, j);
            }
        }
    }

    if (dim == 1) {
        /* Sum over dimension 1 (column): sum each row */
        assert(v->features == M->n);
        for (size_t i = 0; i < M->n; i++) {
            double row_sum = 0;
            for (size_t j = 0; j < M->features; j++) {
                row_sum += matrixGet(M, i, j);
            }
            v->data[i] = row_sum;
        }
    }
}

void tensorAddScaled(Tensor* A, double scale, Tensor* B) {
    assert(A->flat_size == B->flat_size);
    for (size_t i = 0; i < A->flat_size; i++) {
        A->data[i] += scale * B->data[i];
    }
}

void ensureTensorCapacity(Tensor** source, Tensor* reference) {
    bool is_first_time = *source == nullptr;
    size_t current_capacity = is_first_time ? 0 : (*source)->flat_size;
    /* Make sure to have enough memory if the batch size or sequence lengh
     * changes. */
    if (reference->flat_size <= current_capacity) {
        /* FIXME Brittle hack, design better. */
        /* Conform to the posssibly smaller batch size of the ref tensor.*/
        (*source)->batch = reference->batch;
        return;
    }
    if (!is_first_time) {
        /* NOTE destroyTensor already checks that the pointer is not null
         * before calling free, but that's a safety net, do not count on it
         * as a logical feature. */
        destroyTensor(*source);
    }
    *source = zerosLike(reference);
    assert(*source != nullptr);
}

/* Treats the passed tensor as a vector */
size_t argmax(Tensor* t) {
    assert(t != nullptr);

    size_t max_idx = 0;
    double max_value = t->data[max_idx];
    for (size_t i = 0; i < t->flat_size; i++) {
        if (t->data[i] > max_value) {
            max_value = t->data[i];
            max_idx = i;
        }
    }
    return max_idx;
}
