#include <assert.h>

#include "sgd.h"

void sgdStep(SGD optimizer) {
    for (size_t i = 0; i < optimizer.params_count; i++) {
        LearnableParam p = optimizer.params[i];
        tensorAddScaled(p.param, -optimizer.lr, p.grad);
    }
}

SGD createSGD(double lr,
              Tensor* learnable_params[],
              Tensor* grads[],
              size_t params_count) {
    assert(learnable_params != nullptr && grads != nullptr);

    SGD optimizer = {.lr = lr, .params_count = params_count};
    optimizer.params = malloc(params_count * sizeof(LearnableParam));
    if (optimizer.params == nullptr) {
        return optimizer;
    }

    for (size_t i = 0; i < params_count; i++) {
        optimizer.params[i] =
            (LearnableParam){.param = learnable_params[i], .grad = grads[i]};
    }
    optimizer.step = &sgdStep;
    return optimizer;
}

void destroySGD(SGD optimizer) {
    if (optimizer.params != nullptr) {
        free(optimizer.params);
    }
}
