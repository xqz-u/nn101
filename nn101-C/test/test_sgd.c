#ifdef TEST

#include "unity.h"

#include <unistd.h>

#include "linear.h"
#include "nn_math.h"
#include "sgd.h"
#include "tensor.h"
#include "utils.h"

#define SAMPLES 4
#define ROWS 3
#define IN_FEATURES 5
#define OUT_FEATURES 3

static const double input_tensor_array[SAMPLES * ROWS * IN_FEATURES] = {
    3, 5, 5, 3, 2, 8, 8, 3, 8, 4, 7, 8, 9, 2, 4, 2, 1, 8, 1, 5,
    6, 4, 9, 9, 7, 8, 4, 9, 9, 9, 5, 7, 6, 7, 9, 3, 2, 8, 2, 6,
    2, 1, 4, 3, 1, 7, 5, 5, 4, 1, 2, 5, 1, 3, 7, 1, 4, 7, 9, 2,
};

void setUp(void) {}

void tearDown(void) {
    sleep(1);
}

static Tensor* createCommonInput(size_t batch, size_t n, size_t features) {
    Tensor* X = createTensor(batch, n, features);
    size_t input_size = SAMPLES * ROWS * IN_FEATURES;
    size_t limit = X->flat_size > input_size ? input_size : X->flat_size;
    for (size_t i = 0; i < limit; i++) {
        X->data[i] = input_tensor_array[i];
    }
    return X;
}

void test_SGD(void) {
    Tensor* w = onesMatrix(2, 2);

    Tensor* w_grad = zerosLike(w);
    tensorFill(w_grad, 2.0);

    Tensor* params[] = {w};
    Tensor* grads[] = {w_grad};

    /* Edge case with 0 learning rate */
    SGD optim_zero = createSGD(0.0, params, grads, 1);
    optim_zero.step(optim_zero);

    for (size_t i = 0; i < w->flat_size; i++) {
        TEST_ASSERT_EQUAL_DOUBLE(1.0, w->data[i]);
    }
    destroySGD(optim_zero);

    /* Standard Case */
    SGD optim = createSGD(0.1, params, grads, 1);

    TEST_ASSERT_EQUAL_PTR(w, optim.params[0].param);
    TEST_ASSERT_EQUAL_PTR(w_grad, optim.params[0].grad);

    optim.step(optim);

    for (size_t i = 0; i < w->flat_size; i++) {
        TEST_ASSERT_EQUAL_DOUBLE(0.8, w->data[i]);
    }

    destroySGD(optim);
    destroyTensor(w);
    destroyTensor(w_grad);
}

void test_sgdTrainStep(void) {
    gsl_rng* rng = initRNG();

    Tensor* X = createCommonInput(SAMPLES, ROWS, IN_FEATURES);

    Linear* layer = createLinearLayer(IN_FEATURES, OUT_FEATURES);
    layer->init(layer, rng);

    Tensor* Y = createTensor(SAMPLES, ROWS, OUT_FEATURES);

    layer->forward(layer, X, Y);

    Tensor* dLdY = onesLike(Y);
    Tensor* dLdX = zerosLike(X);

    layer->backward(layer, dLdY, dLdX);

    Tensor* params[] = {layer->weights, layer->bias};
    Tensor* grads[] = {layer->weights_grad, layer->bias_grad};
    size_t params_count = sizeof(params) / sizeof(params[0]);

    double lr = 0.01;
    SGD optim = createSGD(lr, params, grads, params_count);

    TEST_ASSERT_EQUAL_PTR(layer->weights, optim.params[0].param);
    TEST_ASSERT_EQUAL_PTR(layer->weights_grad, optim.params[0].grad);
    TEST_ASSERT_EQUAL_PTR(layer->bias, optim.params[1].param);
    TEST_ASSERT_EQUAL_PTR(layer->bias_grad, optim.params[1].grad);

    size_t weights_size = layer->weights->flat_size;
    size_t bias_size = layer->bias->flat_size;

    double weights_copy[weights_size];
    for (size_t i = 0; i < weights_size; i++) {
        weights_copy[i] = layer->weights->data[i];
    }

    double bias_copy[bias_size];
    for (size_t i = 0; i < bias_size; i++) {
        bias_copy[i] = layer->bias->data[i];
    }

    optim.step(optim);

    double expected_new_weights[weights_size];
    for (size_t i = 0; i < weights_size; i++) {
        expected_new_weights[i] =
            weights_copy[i] - lr * layer->weights_grad->data[i];
    }

    TEST_ASSERT_EQUAL_DOUBLE_ARRAY(expected_new_weights, layer->weights->data,
                                   weights_size);

    double expected_new_bias[bias_size];
    for (size_t i = 0; i < bias_size; i++) {
        expected_new_bias[i] = bias_copy[i] - lr * layer->bias_grad->data[i];
    }
    TEST_ASSERT_EQUAL_DOUBLE_ARRAY(expected_new_bias, layer->bias->data,
                                   bias_size);

    destroyTensor(X);
    destroyTensor(Y);
    destroyTensor(dLdY);
    destroyTensor(dLdX);
    destroyLinearLayer(layer);
    destroySGD(optim);
    destroyRNG(rng);
}

#endif  // TEST
