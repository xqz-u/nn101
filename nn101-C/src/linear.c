#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

#include "linear.h"

void forwardLinear(Linear* layer, Tensor* X, Tensor* Y) {
    transpose(layer->weights, layer->weights_T);

    batchedMatMul(X, layer->weights_T, Y);
    addTensorAndVector(Y, layer->bias);

    layer->last_input = X;
}

void backwardLinear(Linear* layer, Tensor* grad_in, Tensor* inputs_grad) {}

Linear* createLinearLayer(size_t in_features, size_t out_features) {
    Linear* layer = malloc(sizeof(Linear));
    if (layer == NULL) {
        return NULL;
    }

    layer->bias = createVector(out_features);
    if (layer->bias == NULL) {
        free(layer);
        return NULL;
    }
    layer->bias_grad = createVector(out_features);
    if (layer->bias_grad == NULL) {
        destroyTensor(layer->bias);
        free(layer);
        return NULL;
    }

    layer->weights = createMatrix(out_features, in_features);
    if (layer->weights == NULL) {
        destroyTensor(layer->bias);
        destroyTensor(layer->bias_grad);
        free(layer);
        return NULL;
    }

    layer->weights_grad = createMatrix(out_features, in_features);
    if (layer->weights_grad == NULL) {
        destroyTensor(layer->weights);
        destroyTensor(layer->bias);
        destroyTensor(layer->bias_grad);
        free(layer);
        return NULL;
    }

    layer->weights_T = createMatrix(in_features, out_features);
    if (layer->weights_T == NULL) {
        destroyTensor(layer->weights);
        destroyTensor(layer->weights_grad);
        destroyTensor(layer->bias);
        destroyTensor(layer->bias_grad);
        free(layer);
        return NULL;
    }

    layer->in_features = in_features;
    layer->out_features = out_features;
    layer->last_input = NULL;
    layer->forward = &forwardLinear;
    layer->backward = &backwardLinear;

    return layer;
}

void destroyLinearLayer(Linear* layer) {
    if (layer == NULL) {
        return;
    }
    destroyTensor(layer->bias);
    destroyTensor(layer->bias_grad);
    destroyTensor(layer->weights);
    destroyTensor(layer->weights_T);
    destroyTensor(layer->weights_grad);
    layer->last_input = NULL;
    layer->forward = NULL;
    layer->backward = NULL;
    free(layer);
}

void initLinearLayer(Linear* layer) {}
