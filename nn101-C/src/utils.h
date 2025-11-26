#ifndef UTILS_H
#define UTILS_H

#define printRealArray(...) var_printRealArray((PrintArgs){__VA_ARGS__});

#include <gsl/gsl_rng.h>

#include "nn_types.h"

typedef struct {
    double* array;
    size_t size;
    const char* fmt;
} PrintArgs;

void var_printRealArray(PrintArgs args);

gsl_rng* initRNG(void);
void destroyRNG(gsl_rng* rng);
double computeMean(double array[], size_t size);
double computeStd(double array[], size_t size);
double relativeError(double a, double b);
void zeroGrads(LearnableParam params[], size_t params_count);
void printIntArray(size_t a[], size_t size);
double max(double* a, size_t size);

#endif  // UTILS_H
