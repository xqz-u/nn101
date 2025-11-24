#include <assert.h>

#include "sgd.h"
#include "utils.h"

void SGDStep(SGD optimizer) {
    for (size_t i = 0; i < optimizer.params_count; i++) {
        LearnableParam p = optimizer.params[i];
        tensorAddScaled(p.param, -optimizer.lr, p.grad);
    }
}

void SGDZeroGrads(SGD optimizer) {
    zeroGrads(optimizer.params, optimizer.params_count);
}

SGD createSGD(double lr, LearnableParam params[], size_t params_count) {
    assert(params != nullptr);

    SGD optimizer = {.lr = lr,
                     .params = nullptr,
                     .params_count = params_count,
                     .step = &SGDStep,
                     .zeroGrads = &SGDZeroGrads};

    LearnableParam* sgdParams = malloc(params_count * sizeof(LearnableParam));
    if (sgdParams == nullptr) {
        return optimizer;
    }
    for (size_t i = 0; i < params_count; i++) {
        sgdParams[i] = params[i];
    }

    optimizer.params = sgdParams;
    return optimizer;
}

void destroySGD(SGD optimizer) {
    if (optimizer.params != nullptr) {
        free(optimizer.params);
    }
}
