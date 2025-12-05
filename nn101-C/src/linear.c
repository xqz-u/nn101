#include <assert.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

#include "linear.h"
#include "nn_math.h"
#include "tensor.h"

void initLinearLayer(Linear* layer, gsl_rng* rng) {
    kaimingUniform(layer->weights->param, rng);
    kaimingUniform(layer->bias->param, rng);
}

void resetGradientsLinear(Linear* layer) {
    tensorFill(layer->weights->grad, 0);
    tensorFill(layer->bias->grad, 0);
}

/* Operates like pytorch: if the input `X` has more than 2 dimensions, only the
 * last is preserved while the others are coalesced into a single one to force a
 * 2D matrix. */
void forwardLinear(Linear* layer, Tensor* X, Tensor* Y) {
    /* Stay safe & prevent gradient accumulation. */
    tensorFill(Y, 0);

    transpose(layer->weights->param, layer->weights_T);
    batchMatMul(X, layer->weights_T, Y);
    addTensorAndVector(Y, layer->bias->param);

    layer->last_input = X;
}

/* @param `grad_out` The incoming gradient, taken wrt to the output of this
 *                   layer.
 * @param `grad_in` The gradient wrt to the input to this layer, to be passed to
 *                  the previous layer. */
void backwardLinear(Linear* layer, Tensor* grad_out, Tensor* grad_in) {
    assert(layer->last_input != nullptr);

    /* Stay safe & prevent gradient accumulation. */
    tensorFill(grad_in, 0);

    /* Rank 2 is forced upon gradients & inputs. Also check the expected shape
     * of the gradient passed in via chain rule. */
    Tensor grad_out_view = {};
    flattenFirstDims(grad_out, &grad_out_view);
    assert(grad_out_view.features == layer->out_features);

    ensureTensorCapacity(&layer->grad_out_T, grad_out);

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
    matMul(&grad_out_T_view, &last_input_view, layer->weights->grad);
    /* dL/db = 1s * dL/dY == sum(dL/dY, axis=0) */
    matrixSum(&grad_out_view, 0, layer->bias->grad);
    /* dL/dX = dL/dY * W */
    batchMatMul(grad_out, layer->weights->param, grad_in);

    layer->last_input = nullptr;
}

Linear* createLinearLayer(size_t in_features, size_t out_features) {
    Linear* layer = malloc(sizeof(Linear));
    if (layer == nullptr) {
        return nullptr;
    }

    layer->weights = createLearnableParam(1, out_features, in_features);
    if (layer->weights == nullptr) {
        destroyLinearLayer(layer);
        return nullptr;
    }

    layer->bias = createLearnableParam(1, 1, out_features);
    if (layer->bias == nullptr) {
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

    destroyLearnableParam(layer->weights);
    destroyLearnableParam(layer->bias);

    if (layer->weights_T != nullptr) {
        destroyTensor(layer->weights_T);
    }
    if (layer->grad_out_T != nullptr) {
        destroyTensor(layer->grad_out_T);
    }

    free(layer);
}
