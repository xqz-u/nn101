#ifndef LINEAR_H
#define LINEAR_H

#include <gsl/gsl_rng.h>

#include "tensor.h"

typedef struct Linear Linear;
struct Linear {
    size_t in_features;
    size_t out_features;

    Tensor* weights;
    Tensor* weights_grad;
    Tensor* bias;
    Tensor* bias_grad;

    Tensor* last_input;

    Tensor* weights_T;
    Tensor* grad_in_T;
    Tensor* ones_like_input;

    void (*forward)(Linear* layer, Tensor* X, Tensor* Y);
    void (*backward)(Linear* layer, Tensor* grad_in, Tensor* inputs_grad);
    void (*setup)(Linear* layer, Tensor* example_input, gsl_rng* rng);
};

Linear* createLinearLayer(size_t in_features, size_t out_features);
void destroyLinearLayer(Linear* layer);

#endif  // LINEAR_H
