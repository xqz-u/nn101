#ifndef UTILS_H
#define UTILS_H

#include <gsl/gsl_rng.h>

gsl_rng* initRNG(void);
double computeMean(double* array, size_t size);
double computeStd(double* array, size_t size);

#endif  // UTILS_H
