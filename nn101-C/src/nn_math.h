#ifndef NN_MATH_H
#define NN_MATH_H

#include <gsl/gsl_rng.h>

#include "tensor.h"

typedef struct Fan Fan;
struct Fan {
    size_t in;
    size_t out;
};

Fan computeFanInFanOut(Tensor* M);
void kaimingUniform(Tensor* weights, gsl_rng* rng);
double logSumExp(double* data, size_t size);

#endif  // NN_MATH_H
