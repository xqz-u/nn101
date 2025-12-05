#ifndef MLP_H
#define MLP_H

#include "activation.h"
#include "linear.h"

typedef struct MLP MLP;
struct MLP {
    size_t n_features;
    size_t* features;  // array
    size_t n_layers;

    Linear** linears;
    ELU activation;

    /* Arrays of struct pointers */
    Tensor** logits;
    Tensor** logits_grad;
    Tensor** activations;
    Tensor** activations_grad;

    void (*forward)(MLP* self, Tensor* X);
    void (*backward)(MLP* self, Tensor* X_grad);
};

MLP createMLP(size_t n_features, size_t features[], gsl_rng* rng);
void destroyMLP(MLP* model);

#endif  // MLP_H
