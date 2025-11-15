#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

#include "vector.h"

Vector* createVector(size_t size) {
    Vector* v = malloc(sizeof(Vector));
    if (v == NULL) {
        return NULL;
    }
    v->data = calloc(size, sizeof(double));
    if (v->data == NULL) {
        free(v);
        return NULL;
    }
    v->size = size;
    return v;
}

void destroyVector(Vector* v) {
    if (v == NULL) {
        return;
    }
    free(v->data);
    free(v);
}

void printRealArray(double* v, size_t size) {
    if (size == 0) {
        printf("[]\n");
        return;
    }
    printf("[%.1f", v[0]);
    for (size_t i = 1; i < size; i++) {
        printf(",%.1f", v[i]);
    }
    printf("]\n");
}

void printVector(Vector* v) {
    printRealArray(v->data, v->size);
}

void fillVector(Vector* v, double* data) {
    for (size_t i = 0; i < v->size; i++) {
        v->data[i] = data[i];
    }
}

double dotProduct(Vector* a, Vector* b) {
    assert(a->size == b->size);
    double z = 0;
    for (size_t i = 0; i < a->size; i++) {
        z += a->data[i] * b->data[i];
    }
    return z;
}
