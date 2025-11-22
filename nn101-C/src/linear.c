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

void resetGradientsLinear(Linear* layer) {
    tensorFill(layer->weights_grad, 0);
    tensorFill(layer->bias_grad, 0);
}

/* Operates like pytorch: if the input `X` has more than 2 dimensions, only the
 * last is preserved while the others are coalesced into a single one to force a
 * 2D matrix. */
void forwardLinear(Linear* layer, Tensor* X, Tensor* Y) {
    /* Ensure not to accumulate output during repeated calls to forward. */
    tensorFill(Y, 0.0);

    transpose(layer->weights, layer->weights_T);
    batchMatMul(X, layer->weights_T, Y);
    addTensorAndVector(Y, layer->bias);

    layer->last_input = X;
}

void ensureVariableTensorCapacity(Tensor** grad_out_T, Tensor* grad_out) {
    bool is_first_time = *grad_out_T == nullptr;
    size_t current_capacity = is_first_time ? 0 : (*grad_out_T)->flat_size;

    /* Make sure to have enough memory if the batch size or sequence lengh
     * (grad_out->n) changes. */
    if (grad_out->flat_size <= current_capacity) {
        return;
    }
    if (!is_first_time) {
        /* NOTE destroyTensor already checks that the pointer is not null
         * before calling free, but that's a safety net, do not count on it
         * as a logical feature. */
        destroyTensor(*grad_out_T);
    }
    *grad_out_T =
        createTensor(grad_out->batch, grad_out->features, grad_out->n);
}

/* @param `grad_out` The incoming gradient, taken wrt to the output of this
 *                   layer.
 * @param `grad_in` The gradient wrt to the input to this layer, to be passed to
 *                  the previous layer. */
void backwardLinear(Linear* layer, Tensor* grad_out, Tensor* grad_in) {
    assert(layer->last_input != nullptr);

    layer->reset(layer);

    /* Rank 2 is forced upon gradients & inputs. Also check the expected shape
     * of the gradient passed in via chain rule. */
    Tensor grad_out_view = {};
    flattenFirstDims(grad_out, &grad_out_view);
    assert(grad_out_view.features == layer->out_features);

    ensureVariableTensorCapacity(&layer->grad_out_T, grad_out);

    Tensor grad_out_T_view = {};
    flattenFirstDims(layer->grad_out_T, &grad_out_T_view);
    /* After downcasting to rank 2, we can then take the transpose with the
     * right dimensionality obtained from the cast. */
    grad_out_T_view.n = grad_out_view.features;
    grad_out_T_view.features = grad_out_view.n;

    transpose(&grad_out_view, &grad_out_T_view);

    Tensor last_input_view = {};
    flattenFirstDims(layer->last_input, &last_input_view);

    /* dL/dW = (dL/dY)^T * X */
    matMul(&grad_out_T_view, &last_input_view, layer->weights_grad);
    /* dL/db = 1s * dL/dY == sum(dL/dY, axis=0) */
    matrixSum(&grad_out_view, 0, layer->bias_grad);
    /* dL/dX = dL/dY * W */
    batchMatMul(grad_out, layer->weights, grad_in);

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

    layer->init = &initLinearLayer;
    layer->forward = &forwardLinear;
    layer->backward = &backwardLinear;
    layer->reset = &resetGradientsLinear;

    layer->in_features = in_features;
    layer->out_features = out_features;
    layer->last_input = nullptr;
    layer->grad_out_T = nullptr;

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
    if (layer->grad_out_T != nullptr) {
        destroyTensor(layer->grad_out_T);
    }

    free(layer);
}
