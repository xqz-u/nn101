#ifndef LINEAR_H
#define LINEAR_H

#include <gsl/gsl_rng.h>

#include "nn_types.h"
#include "tensor.h"

// TODO comment shape semantics
typedef struct Linear Linear;
struct Linear {
    size_t in_features;
    size_t out_features;

    LearnableParam weights;
    LearnableParam bias;

    Tensor* last_input;

    Tensor* weights_T;
    Tensor* grad_out_T;

    void (*forward)(Linear* layer, Tensor* X, Tensor* Y);
    void (*backward)(Linear* layer, Tensor* grad_out, Tensor* grad_in);
    void (*init)(Linear* layer, gsl_rng* rng);
};

Linear* createLinearLayer(size_t in_features, size_t out_features);
void destroyLinearLayer(Linear* layer);

#endif  // LINEAR_H
