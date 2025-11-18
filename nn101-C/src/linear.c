#include <assert.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

#include "linear.h"
#include "nn_math.h"

void initLinearLayer(Linear* layer, gsl_rng* rng) {
    kaimingUniform(layer->weights, rng);
    kaimingUniform(layer->bias, rng);
}

void buildComputationGraphLinear(Linear* layer, Tensor* example_input) {
    assert(example_input->batch == 1);

    assert(layer->grad_in_T == NULL);
    layer->grad_in_T = tensorLike(example_input);
    assert(layer->grad_in_T != NULL);

    assert(layer->ones_like_input == NULL);
    layer->ones_like_input = onesMatrix(1, example_input->n);
    assert(layer->ones_like_input != NULL);
}

void setupLinear(Linear* layer, Tensor* example_input, gsl_rng* rng) {
    initLinearLayer(layer, rng);
    buildComputationGraphLinear(layer, example_input);
}

void forwardLinear(Linear* layer, Tensor* X, Tensor* Y) {
    assert(layer->last_input == NULL);

    transpose(layer->weights, layer->weights_T);
    batchedMatMul(X, layer->weights_T, Y);
    addTensorAndVector(Y, layer->bias);

    layer->last_input = X;
}

void backwardLinear(Linear* layer, Tensor* grad_in, Tensor* inputs_grad) {
    assert(layer->last_input != NULL);

    transpose(grad_in, layer->grad_in_T);
    matMul(layer->grad_in_T, layer->last_input, layer->weights_grad);
    matMul(layer->ones_like_input, grad_in, layer->bias_grad);

    matMul(grad_in, layer->weights, inputs_grad);

    layer->last_input = NULL;
}

Linear* createLinearLayer(size_t in_features, size_t out_features) {
    Linear* layer = malloc(sizeof(Linear));
    if (layer == NULL) {
        return NULL;
    }

    layer->bias = createVector(out_features);
    if (layer->bias == NULL) {
        destroyLinearLayer(layer);
        return NULL;
    }
    layer->bias_grad = tensorLike(layer->bias);
    if (layer->bias_grad == NULL) {
        destroyLinearLayer(layer);
        return NULL;
    }

    layer->weights = createMatrix(out_features, in_features);
    if (layer->weights == NULL) {
        destroyLinearLayer(layer);
        return NULL;
    }

    layer->weights_grad = tensorLike(layer->weights);
    if (layer->weights_grad == NULL) {
        destroyLinearLayer(layer);
        return NULL;
    }

    layer->weights_T = createMatrix(in_features, out_features);
    if (layer->weights_T == NULL) {
        destroyLinearLayer(layer);
        return NULL;
    }

    layer->forward = &forwardLinear;
    layer->backward = &backwardLinear;
    layer->setup = &setupLinear;

    layer->in_features = in_features;
    layer->out_features = out_features;
    layer->last_input = NULL;
    layer->grad_in_T = NULL;
    layer->ones_like_input = NULL;

    return layer;
}

void destroyLinearLayer(Linear* layer) {
    if (layer == NULL) {
        return;
    }

    if (layer->bias != NULL) {
        destroyTensor(layer->bias);
    }
    if (layer->bias_grad != NULL) {
        destroyTensor(layer->bias_grad);
    }
    if (layer->weights != NULL) {
        destroyTensor(layer->weights);
    }
    if (layer->weights_grad != NULL) {
        destroyTensor(layer->weights_grad);
    }
    if (layer->weights_T != NULL) {
        destroyTensor(layer->weights_T);
    }
    if (layer->grad_in_T != NULL) {
        destroyTensor(layer->grad_in_T);
    }
    if (layer->ones_like_input != NULL) {
        destroyTensor(layer->ones_like_input);
    }

    layer->forward = NULL;
    layer->backward = NULL;
    layer->last_input = NULL;

    free(layer);
}
