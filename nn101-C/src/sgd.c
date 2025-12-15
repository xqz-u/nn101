#include <assert.h>
#include <string.h>

#include "learnable_param.h"
#include "sgd.h"
#include "utils.h"

void SGDStep(SGD* optimizer) {
    for (size_t i = 0; i < optimizer->params_count; i++) {
        LearnableParam* p = optimizer->params[i];
        tensorAddScaled(p->param, -optimizer->lr, p->grad);
    }
}

void SGDZeroGrads(SGD* optimizer) {
    zeroGrads(optimizer->params, optimizer->params_count);
}

SGD createSGD(double lr, LearnableParam* params[], size_t params_count) {
    assert(params != nullptr);

    SGD optimizer = {.lr = lr,
                     .params = nullptr,
                     .params_count = params_count,
                     .step = &SGDStep,
                     .zeroGrads = &SGDZeroGrads};

    size_t learnable_params_size = params_count * sizeof(LearnableParam*);

    LearnableParam** sgdParams = malloc(learnable_params_size);
    if (sgdParams == nullptr) {
        return optimizer;
    }
    memcpy(sgdParams, params, learnable_params_size);

    optimizer.params = sgdParams;
    return optimizer;
}

SGD createSGDFromModel(double lr, MLP* model) {
    /* weights matrix + bias vector for each block */
    size_t tot_learnable_params = model->n_layers * 2;
    LearnableParam* params[tot_learnable_params];

    size_t p_idx = 0;
    for (size_t i = 0; i < model->n_layers; i++) {
        Linear* layer = model->linears[i];
        params[p_idx++] = layer->weights;
        params[p_idx++] = layer->bias;
    }

    return createSGD(lr, params, tot_learnable_params);
}

void destroySGD(SGD* optimizer) {
    if (optimizer->params == nullptr) {
        return;
    }
    free(optimizer->params);
}
