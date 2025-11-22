#ifndef FINITE_DIFF_H
#define FINITE_DIFF_H

#include "linear.h"

double finiteDiffLinearWeights(Linear* layer,
                               Tensor* X,
                               double eps,
                               gsl_rng* rng);

double finiteDiffLinearBias(Linear* layer, Tensor* X, double eps, gsl_rng* rng);

double finiteDiffLinearInput(Linear* layer,
                             Tensor* X,
                             double eps,
                             gsl_rng* rng);

#endif  // FINITE_DIFF
