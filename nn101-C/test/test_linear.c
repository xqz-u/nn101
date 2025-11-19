#ifdef TEST

#include "unity.h"

#include <gsl/gsl_rng.h>
#include <unistd.h>

#include "linear.h"
#include "nn_math.h"
#include "tensor.h"
#include "utils.h"

#define SAMPLES 4
#define ROWS 3
#define IN_FEATURES 5
#define OUT_FEATURES 3

static gsl_rng* rng = NULL;

static const double input_tensor_array[SAMPLES * ROWS * IN_FEATURES] = {
    3, 5, 5, 3, 2,
    //
    8, 8, 3, 8, 4,
    //
    7, 8, 9, 2, 4,
    /*  */
    2, 1, 8, 1, 5,
    //
    6, 4, 9, 9, 7,
    //
    8, 4, 9, 9, 9,
    /*  */
    5, 7, 6, 7, 9,
    //
    3, 2, 8, 2, 6,
    //
    2, 1, 4, 3, 1,
    /*  */
    7, 5, 5, 4, 1,
    //
    2, 5, 1, 3, 7,
    //
    1, 4, 7, 9, 2};

void setUp(void) {
    rng = initRNG();
}

void tearDown(void) {
    gsl_rng_free(rng);
    sleep(1);
}

void test_linearForward(void) {
    /* initialize input */
    Tensor* X = createTensor(SAMPLES, ROWS, IN_FEATURES);
    for (size_t i = 0; i < X->flat_size; i++) {
        X->data[i] = input_tensor_array[i];
    }
    /* printTensor(X); */

    Linear* layer = createLinearLayer(IN_FEATURES, OUT_FEATURES);

    /* bias vector of 1 */
    destroyTensor(layer->bias);
    layer->bias = onesVector(OUT_FEATURES);
    /* printVector(layer->bias); */

    /* initialize output */
    Tensor* Y = createTensor(SAMPLES, ROWS, OUT_FEATURES);

    /* adjust the expected output to account for the bias vector */
    double expected[SAMPLES * ROWS * OUT_FEATURES] = {18, 36, 54,
                                                      //
                                                      31, 62, 93,
                                                      //
                                                      30, 60, 90,
                                                      /*  */
                                                      17, 34, 51,
                                                      //
                                                      35, 70, 105,
                                                      //
                                                      39, 78, 117,
                                                      /*  */
                                                      34, 68, 102,
                                                      //
                                                      21, 42, 63,
                                                      //
                                                      11, 22, 33,
                                                      /*  */
                                                      22, 44, 66,
                                                      //
                                                      18, 36, 54,
                                                      //
                                                      23, 46, 69};
    for (size_t i = 0; i < Y->flat_size; i++) {
        expected[i] += layer->bias->data[0];
    }

    /* initialize the weights */
    for (size_t i = 1; i < OUT_FEATURES + 1; i++) {
        for (size_t j = 0; j < IN_FEATURES; j++) {
            size_t idx = indexMatrix(layer->weights, i - 1, j);
            layer->weights->data[idx] = (double)i;
        }
    }

    /* NOTE Passing the input like so is not the classic dense layer way, where
     * every entry in a single 2D input participates in the same linear
     * transformation. In this scenario (batch, dim0, dim1) each row of a single
     * 2D input is linearly transformed *separately*; batch and dim0 should be
     * flattened in the conventional fully connected MLP approach. */
    layer->forward(layer, X, Y);
    /* printTensor(Y); */

    for (size_t i = 0; i < Y->flat_size; i++) {
        TEST_ASSERT_EQUAL_DOUBLE(Y->data[i], expected[i]);
    }

    destroyTensor(X);
    destroyTensor(Y);
    destroyLinearLayer(layer);
}

void test_linearBackward(void) {
    size_t flat_features_in = ROWS * IN_FEATURES;
    size_t flat_features_out = ROWS * OUT_FEATURES;

    Tensor* X = createMatrix(SAMPLES, flat_features_in);
    for (size_t i = 0; i < X->flat_size; i++) {
        X->data[i] = input_tensor_array[i];
    }

    Tensor* Y = createMatrix(SAMPLES, flat_features_out);

    Linear* layer = createLinearLayer(flat_features_in, flat_features_out);

    /* To check the correctness of the math, we control the learnable params
     * default initialization */
    layer->setup(layer, X, rng);

    /* Below weights are transposed (same happens inside the linear layer).
     */
    /* Shape: flat_features_in * flat_features_out */
    double weights_T[] = {
        4, 5, 5, 5, 7, 5, 6, 2, 9, 3, 5, 8, 4, 6, 6, 5, 3, 9, 5, 6, 9, 5, 8,
        2, 7, 1, 6, 3, 8, 4, 9, 9, 2, 6, 4, 4, 1, 1, 7, 2, 1, 5, 1, 3, 4, 9,
        5, 6, 2, 8, 6, 5, 9, 8, 7, 8, 8, 1, 3, 8, 9, 8, 7, 7, 3, 7, 3, 1, 2,
        6, 1, 3, 9, 7, 8, 7, 7, 7, 1, 9, 9, 9, 3, 2, 5, 4, 1, 6, 3, 7, 4, 5,
        6, 9, 8, 4, 9, 4, 2, 6, 6, 7, 2, 8, 2, 9, 1, 3, 5, 5, 1, 7, 8, 6, 5,
        8, 7, 5, 1, 6, 9, 2, 5, 9, 5, 9, 4, 7, 6, 2, 6, 1, 1, 5, 3};
    for (size_t i = 0; i < layer->weights->flat_size; i++) {
        layer->weights->data[i] = weights_T[i];
    }

    /* Shape: flat_features_out */
    double bias[] = {9, 3, 1, 5, 8, 6, 9, 2, 1};
    for (size_t i = 0; i < layer->bias->flat_size; i++) {
        layer->bias->data[i] = bias[i];
    }

    layer->forward(layer, X, Y);

    /* Shape: SAMPLES * flat_features_out */
    double expected_Y[] = {449, 399, 294, 483, 418, 431, 415, 413, 362,
                           492, 503, 284, 580, 471, 479, 459, 470, 429,
                           329, 387, 271, 410, 372, 372, 390, 327, 348,
                           335, 356, 259, 348, 410, 336, 365, 300, 285};
    TEST_ASSERT_EQUAL_DOUBLE_ARRAY(expected_Y, Y->data, Y->flat_size);

    /* Imagine that the loss is just the sum of the output: the gradient wrt
to
     * the loss is then just a 1s tensor. */
    /* Shape: same as output */
    Tensor* dLdY = onesLike(Y);

    /* Shape: same as input */
    Tensor* dLdX = zerosLike(X);

    layer->backward(layer, dLdY, dLdX);

    /* Shape: same as layer->weight */
    double expected_dLdW[] = {
        17, 18, 24, 15, 17, 19, 19, 21, 22, 24, 18, 17, 29, 23, 16, 17, 18,
        24, 15, 17, 19, 19, 21, 22, 24, 18, 17, 29, 23, 16, 17, 18, 24, 15,
        17, 19, 19, 21, 22, 24, 18, 17, 29, 23, 16, 17, 18, 24, 15, 17, 19,
        19, 21, 22, 24, 18, 17, 29, 23, 16, 17, 18, 24, 15, 17, 19, 19, 21,
        22, 24, 18, 17, 29, 23, 16, 17, 18, 24, 15, 17, 19, 19, 21, 22, 24,
        18, 17, 29, 23, 16, 17, 18, 24, 15, 17, 19, 19, 21, 22, 24, 18, 17,
        29, 23, 16, 17, 18, 24, 15, 17, 19, 19, 21, 22, 24, 18, 17, 29, 23,
        16, 17, 18, 24, 15, 17, 19, 19, 21, 22, 24, 18, 17, 29, 23, 16};

    TEST_ASSERT_EQUAL_DOUBLE_ARRAY(expected_dLdW, layer->weights_grad->data,
                                   layer->weights_grad->flat_size);

    /* Shape: flat_features_out */
    double expected_dLdb[] = {4, 4, 4, 4, 4, 4, 4, 4, 4};
    TEST_ASSERT_EQUAL_DOUBLE_ARRAY(expected_dLdb, layer->bias_grad->data,
                                   layer->bias_grad->flat_size);

    /* Shape: same as input */
    double expected_dLdX[] = {65, 45, 50, 49, 55, 54, 49, 43, 44, 43, 40, 46,
                              32, 44, 48, 65, 45, 50, 49, 55, 54, 49, 43, 44,
                              43, 40, 46, 32, 44, 48, 65, 45, 50, 49, 55, 54,
                              49, 43, 44, 43, 40, 46, 32, 44, 48, 65, 45, 50,
                              49, 55, 54, 49, 43, 44, 43, 40, 46, 32, 44, 48};
    TEST_ASSERT_EQUAL_DOUBLE_ARRAY(expected_dLdX, dLdX->data, dLdX->flat_size);

    destroyTensor(dLdX);
    destroyTensor(dLdY);
    destroyTensor(X);
    destroyTensor(Y);
    destroyLinearLayer(layer);
}

void test_linearBackward3d(void) {
    Tensor* X = createTensor(SAMPLES, ROWS, IN_FEATURES);
    for (size_t i = 0; i < X->flat_size; i++) {
        X->data[i] = input_tensor_array[i];
    }

    Tensor* Y = createTensor(SAMPLES, ROWS, OUT_FEATURES);

    Linear* layer = createLinearLayer(IN_FEATURES, OUT_FEATURES);

    /* To check the correctness of the math, we control the learnable params
     * default initialization */
    layer->setup(layer, X, rng);

    /* Below weights are transposed (same happens inside the linear layer). */
    /* Shape: features_in * features_out */
    double weights_T[OUT_FEATURES * IN_FEATURES] = {2, 3, 2, 4, 2, 3, 4, 3,
                                                    6, 9, 2, 8, 1, 1, 1};
    for (size_t i = 0; i < layer->weights->flat_size; i++) {
        layer->weights->data[i] = weights_T[i];
    }

    /* Shape: features_out */
    double bias[] = {2, 5, 8};
    for (size_t i = 0; i < layer->bias->flat_size; i++) {
        layer->bias->data[i] = bias[i];
    }

    layer->forward(layer, X, Y);

    double expected_Y[SAMPLES * ROWS * OUT_FEATURES] = {
        49, 85,  64, 88,  154, 103, 74, 133, 101, 39, 90,  34,
        94, 183, 77, 102, 207, 83,  91, 189, 96,  50, 112, 46,
        31, 54,  28, 59,  94,  72,  49, 115, 63,  70, 117, 60};
    TEST_ASSERT_EQUAL_DOUBLE_ARRAY(expected_Y, Y->data, Y->flat_size);

    /* Imagine that the loss is just the sum of the output: the gradient wrt to
     * the loss is then just a 1s tensor. */
    Tensor* dLdY = onesLike(Y);

    Tensor* dLdX = zerosLike(X);

    layer->backward(layer, dLdY, dLdX);

    double expected_dLdW[OUT_FEATURES * IN_FEATURES] = {
        54, 54, 74, 60, 57, 54, 54, 74, 60, 57, 54, 54, 74, 60, 57};

    TEST_ASSERT_EQUAL_DOUBLE_ARRAY(expected_dLdW, layer->weights_grad->data,
                                   layer->weights_grad->flat_size);

    double expected_dLdb[OUT_FEATURES] = {12, 12, 12};
    TEST_ASSERT_EQUAL_DOUBLE_ARRAY(expected_dLdb, layer->bias_grad->data,
                                   layer->bias_grad->flat_size);

    double expected_dLdX[SAMPLES * ROWS * IN_FEATURES] = {
        7, 15, 6, 11, 12, 7, 15, 6, 11, 12, 7, 15, 6, 11, 12, 7, 15, 6, 11, 12,
        7, 15, 6, 11, 12, 7, 15, 6, 11, 12, 7, 15, 6, 11, 12, 7, 15, 6, 11, 12,
        7, 15, 6, 11, 12, 7, 15, 6, 11, 12, 7, 15, 6, 11, 12, 7, 15, 6, 11, 12};
    TEST_ASSERT_EQUAL_DOUBLE_ARRAY(expected_dLdX, dLdX->data, dLdX->flat_size);

    destroyTensor(dLdX);
    destroyTensor(dLdY);
    destroyTensor(X);
    destroyTensor(Y);
    destroyLinearLayer(layer);
}

#endif  // TEST
