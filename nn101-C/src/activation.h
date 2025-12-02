#ifndef ACTIVATION_H
#define ACTIVATION_H

#include "tensor.h"

typedef struct ELU ELU;
struct ELU {
    double alpha;

    void (*forward)(ELU* self, Tensor* logits, Tensor* activations);
    void (*backward)(ELU* self,
                     Tensor* activations,
                     Tensor* grad_out,
                     Tensor* grad_in);
};

ELU createELUActivation(double alpha);

#endif  // ACTIVATION_H
