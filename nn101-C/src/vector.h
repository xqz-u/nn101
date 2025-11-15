#ifndef VECTOR_H
#define VECTOR_H

#include <stddef.h>

typedef struct Vector Vector;
struct Vector {
    double* data;
    size_t size;
};

Vector* createVector(size_t size);
void destroyVector(Vector* v);
void fillVector(Vector* v, double* data);

void printRealArray(double* v, size_t size);
void printVector(Vector* v);

double dotProduct(Vector* a, Vector* b);

#endif
