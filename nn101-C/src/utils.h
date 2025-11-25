#ifndef UTILS_H
#define UTILS_H

#include <gsl/gsl_rng.h>

#include "nn_types.h"

gsl_rng* initRNG(void);
void destroyRNG(gsl_rng* rng);
double computeMean(double* array, size_t size);
double computeStd(double* array, size_t size);
double relativeError(double a, double b);
void zeroGrads(LearnableParam params[], size_t params_count);
void printRealArray(double* a, size_t size);

#endif  // UTILS_H
