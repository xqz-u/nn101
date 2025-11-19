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
    assert(layer->grad_in_T == nullptr);
    layer->grad_in_T = createTensor(example_input->batch, layer->out_features,
                                    example_input->n);
    assert(layer->grad_in_T != nullptr);

    assert(layer->ones_like_input == nullptr);
    Tensor example_input_view = {};
    flattenFirstDims(example_input, &example_input_view);
    layer->ones_like_input = onesVector(example_input_view.n);
    assert(layer->ones_like_input != nullptr);
}

void setupLinear(Linear* layer, Tensor* example_input, gsl_rng* rng) {
    initLinearLayer(layer, rng);
    buildComputationGraphLinear(layer, example_input);
}

/* Operates like pytorch: if the input `X` has more than 2 dimensions, only the
 * last is preserved while the others are coalesced into a single one to force a
 * 2D matrix. */
void forwardLinear(Linear* layer, Tensor* X, Tensor* Y) {
    assert(layer->last_input == nullptr);

    transpose(layer->weights, layer->weights_T);
    batchMatMul(X, layer->weights_T, Y);
    addTensorAndVector(Y, layer->bias);

    layer->last_input = X;
}

void backwardLinear(Linear* layer, Tensor* grad_in, Tensor* inputs_grad) {
    assert(layer->last_input != nullptr);

    /* Rank 2 is forced upon gradients & inputs. Also check the expected shape
     * of the gradient passed in via chain rule. */
    Tensor grad_in_view = {};
    flattenFirstDims(grad_in, &grad_in_view);
    assert(grad_in_view.features == layer->out_features);

    Tensor grad_in_T_view = {};
    flattenFirstDims(layer->grad_in_T, &grad_in_T_view);
    /* After downcasting to rank 2, we can then take the transpose with the
     * right dimensionality obtained from the cast. */
    grad_in_T_view.n = grad_in_view.features;
    grad_in_T_view.features = grad_in_view.n;

    transpose(&grad_in_view, &grad_in_T_view);

    Tensor last_input_view = {};
    flattenFirstDims(layer->last_input, &last_input_view);

    batchMatMul(&grad_in_T_view, &last_input_view, layer->weights_grad);
    batchMatMul(layer->ones_like_input, &grad_in_view, layer->bias_grad);

    batchMatMul(grad_in, layer->weights, inputs_grad);

    layer->last_input = nullptr;
}

Linear* createLinearLayer(size_t in_features, size_t out_features) {
    Linear* layer = malloc(sizeof(Linear));
    if (layer == nullptr) {
        return nullptr;
    }

    layer->bias = createVector(out_features);
    if (layer->bias == nullptr) {
        destroyLinearLayer(layer);
        return nullptr;
    }
    layer->bias_grad = zerosLike(layer->bias);
    if (layer->bias_grad == nullptr) {
        destroyLinearLayer(layer);
        return nullptr;
    }

    layer->weights = createMatrix(out_features, in_features);
    if (layer->weights == nullptr) {
        destroyLinearLayer(layer);
        return nullptr;
    }

    layer->weights_grad = zerosLike(layer->weights);
    if (layer->weights_grad == nullptr) {
        destroyLinearLayer(layer);
        return nullptr;
    }

    layer->weights_T = createMatrix(in_features, out_features);
    if (layer->weights_T == nullptr) {
        destroyLinearLayer(layer);
        return nullptr;
    }

    layer->forward = &forwardLinear;
    layer->backward = &backwardLinear;
    layer->setup = &setupLinear;

    layer->in_features = in_features;
    layer->out_features = out_features;
    layer->last_input = nullptr;
    layer->grad_in_T = nullptr;
    layer->ones_like_input = nullptr;

    return layer;
}

void destroyLinearLayer(Linear* layer) {
    if (layer == nullptr) {
        return;
    }

    if (layer->bias != nullptr) {
        destroyTensor(layer->bias);
    }
    if (layer->bias_grad != nullptr) {
        destroyTensor(layer->bias_grad);
    }
    if (layer->weights != nullptr) {
        destroyTensor(layer->weights);
    }
    if (layer->weights_grad != nullptr) {
        destroyTensor(layer->weights_grad);
    }
    if (layer->weights_T != nullptr) {
        destroyTensor(layer->weights_T);
    }
    if (layer->grad_in_T != nullptr) {
        destroyTensor(layer->grad_in_T);
    }
    if (layer->ones_like_input != nullptr) {
        destroyTensor(layer->ones_like_input);
    }

    free(layer);
}
