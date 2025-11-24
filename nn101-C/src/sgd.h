#ifndef SGD_H
#define SGD_H

#include "tensor.h"

typedef struct LearnableParam LearnableParam;
struct LearnableParam {
    const char* name;
    Tensor* param;
    Tensor* grad;
};

typedef struct SGD SGD;
struct SGD {
    double lr;
    size_t params_count;
    LearnableParam* params;

    void (*step)(SGD optimizer);
};

SGD createSGD(double lr,
              Tensor* learnable_params[],
              Tensor* grads[],
              size_t params_count);
void destroySGD(SGD optimizer);

#endif  // SGD_H
