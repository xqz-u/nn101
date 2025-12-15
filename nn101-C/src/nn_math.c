#include <assert.h>
#include <gsl/gsl_randist.h>
#include <math.h>

#include "nn_math.h"
#include "utils.h"

/* Simplification of the linked function for the 2D case */
/* https://github.com/pytorch/pytorch/blob/9ff95f6835e839f7a9fc35ebd5757b955740d310/torch/nn/init.py#L417
 */
Fan computeFanInFanOut(Tensor* M) {
    assert(M->batch == 1);
    return (Fan){.in = M->features, .out = M->n};
}

/* Reference: */
/* https://github.com/pytorch/pytorch/blob/9ff95f6835e839f7a9fc35ebd5757b955740d310/torch/nn/init.py#L513
 */
/* NOTE https://github.com/pytorch/pytorch/issues/57109#issuecomment-1310430244
 */
/* NOTE assumes nonlinearity is Leaky RELU */
void kaimingUniform(Tensor* T, gsl_rng* rng) {
    size_t fan_in = computeFanInFanOut(T).in;
    double bound = 1 / sqrt(fan_in);
    randUniformTensor(T, bound, rng);
}

double logSumExp(double* data, size_t size) {
    /* Subtract the max of data from each etry to force e^x to be at max 1 thus
     * avoiding overflow / nans.
     * ref: https://gregorygundersen.com/blog/2020/02/09/log-sum-exp/
     */
    double max_val = max(data, size);

    double sum = 0.0;
    for (size_t i = 0; i < size; i++) {
        sum += exp(data[i] - max_val);
    }
    /* Restore after shifting */
    return log(sum) + max_val;
}
