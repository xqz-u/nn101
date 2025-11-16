#ifndef LINEAR_H
#define LINEAR_H

#include <stddef.h>

#include "tensor.h"
#include "vector.h"

typedef struct Linear Linear;
struct Linear {
    size_t in_features;
    size_t out_features;

    Matrix* weights;
    Matrix* weights_T;
    Matrix* weights_grad;
    Vector* bias;
    Vector* bias_grad;

    Tensor* last_input;

    void (*forward)(Linear* self, Tensor* X, Tensor* Y);
    void (*backward)(Linear* self, Tensor* dLdX, Tensor* Y);
};

Linear* createLinearLayer(size_t in_features, size_t out_features);
void destroyLinearLayer(Linear* layer);
void initLinearLayer(Linear* layer);

#endif  // LINEAR_H
