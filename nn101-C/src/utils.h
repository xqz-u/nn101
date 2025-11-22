#ifndef UTILS_H
#define UTILS_H

#include <gsl/gsl_rng.h>

gsl_rng* initRNG(void);
void destroyRNG(gsl_rng* rng);
double computeMean(double* array, size_t size);
double computeStd(double* array, size_t size);
double relativeError(double a, double b);

#endif  // UTILS_H
