#ifdef TEST

#include "unity.h"

#include <unistd.h>

#include "finite_diff.h"
#include "learnable_param.h"
#include "linear.h"
#include "nn_math.h"
#include "tensor.h"
#include "utils.h"

#define SAMPLES 4
#define ROWS 3
#define IN_FEATURES 5
#define OUT_FEATURES 3

void setUp(void) {}

void tearDown(void) {
    sleep(1);
}

static const double input_tensor_array[SAMPLES * ROWS * IN_FEATURES] = {
    3, 5, 5, 3, 2, 8, 8, 3, 8, 4, 7, 8, 9, 2, 4, 2, 1, 8, 1, 5,
    6, 4, 9, 9, 7, 8, 4, 9, 9, 9, 5, 7, 6, 7, 9, 3, 2, 8, 2, 6,
    2, 1, 4, 3, 1, 7, 5, 5, 4, 1, 2, 5, 1, 3, 7, 1, 4, 7, 9, 2,
};

static Tensor* createCommonInput(size_t batch, size_t n, size_t features) {
    Tensor* X = createTensor(batch, n, features);
    size_t input_size = SAMPLES * ROWS * IN_FEATURES;
    size_t limit = X->flat_size > input_size ? input_size : X->flat_size;
    for (size_t i = 0; i < limit; i++) {
        X->data[i] = input_tensor_array[i];
    }
    return X;
}

static Linear* createCommonLinearLayer(void) {
    size_t flat_features_in = ROWS * IN_FEATURES;
    size_t flat_features_out = ROWS * OUT_FEATURES;

    Linear* layer = createLinearLayer(flat_features_in, flat_features_out);
    /* No need to randomly initialize the weights, they are controlled in these
     * tests */

    /* To check the correctness of the math, we control the learnable params
     * default initialization. */
    /* Below weights are transposed (same happens inside the linear layer). */
    /* Shape: flat_features_in * flat_features_out */
    double weights_T[] = {
        4, 5, 5, 5, 7, 5, 6, 2, 9, 3, 5, 8, 4, 6, 6, 5, 3, 9, 5, 6, 9, 5, 8,
        2, 7, 1, 6, 3, 8, 4, 9, 9, 2, 6, 4, 4, 1, 1, 7, 2, 1, 5, 1, 3, 4, 9,
        5, 6, 2, 8, 6, 5, 9, 8, 7, 8, 8, 1, 3, 8, 9, 8, 7, 7, 3, 7, 3, 1, 2,
        6, 1, 3, 9, 7, 8, 7, 7, 7, 1, 9, 9, 9, 3, 2, 5, 4, 1, 6, 3, 7, 4, 5,
        6, 9, 8, 4, 9, 4, 2, 6, 6, 7, 2, 8, 2, 9, 1, 3, 5, 5, 1, 7, 8, 6, 5,
        8, 7, 5, 1, 6, 9, 2, 5, 9, 5, 9, 4, 7, 6, 2, 6, 1, 1, 5, 3};
    for (size_t i = 0; i < layer->weights->param->flat_size; i++) {
        layer->weights->param->data[i] = weights_T[i];
    }

    /* Shape: flat_features_out */
    double bias[] = {9, 3, 1, 5, 8, 6, 9, 2, 1};
    for (size_t i = 0; i < layer->bias->param->flat_size; i++) {
        layer->bias->param->data[i] = bias[i];
    }

    return layer;
}

static void linearLayerFullPass(Linear* layer,
                                Tensor* X,
                                Tensor* Y,
                                double expected_Y[],
                                double expected_dLdW[],
                                double expected_dLdX[]) {
    zeroGrads((LearnableParam*[]){layer->weights, layer->bias}, 2);

    layer->forward(layer, X, Y);

    TEST_ASSERT_EQUAL_DOUBLE_ARRAY(expected_Y, Y->data, Y->flat_size);

    /* Imagine that the loss is just the sum of the output: the gradient wrt to
     * the loss is then just a 1s tensor. */
    /* Shape: same as output */
    Tensor* dLdY = onesLike(Y);

    /* Shape: same as input */
    Tensor* dLdX = zerosLike(X);

    layer->backward(layer, dLdY, dLdX);

    TEST_ASSERT_EQUAL_DOUBLE_ARRAY(expected_dLdW, layer->weights->grad->data,
                                   layer->weights->grad->flat_size);

    /* NOTE Semplification derived by assumption in above comment. */
    /* Shape: flat_features_out */
    double samples = X->n * X->batch;
    TEST_ASSERT_EACH_EQUAL_DOUBLE(samples, layer->bias->grad->data,
                                  layer->bias->grad->flat_size);

    TEST_ASSERT_EQUAL_DOUBLE_ARRAY(expected_dLdX, dLdX->data, dLdX->flat_size);

    destroyTensor(dLdY);
    destroyTensor(dLdX);
}

static void linearLayerFullPassNormalBatch(Linear* layer) {
    Tensor* X = createCommonInput(1, SAMPLES, layer->in_features);

    Tensor* Y = createMatrix(SAMPLES, layer->out_features);

    /* Shape: SAMPLES * flat_features_out */
    double expected_Y[] = {449, 399, 294, 483, 418, 431, 415, 413, 362,
                           492, 503, 284, 580, 471, 479, 459, 470, 429,
                           329, 387, 271, 410, 372, 372, 390, 327, 348,
                           335, 356, 259, 348, 410, 336, 365, 300, 285};

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

    /* Shape: same as input */
    double expected_dLdX[] = {65, 45, 50, 49, 55, 54, 49, 43, 44, 43, 40, 46,
                              32, 44, 48, 65, 45, 50, 49, 55, 54, 49, 43, 44,
                              43, 40, 46, 32, 44, 48, 65, 45, 50, 49, 55, 54,
                              49, 43, 44, 43, 40, 46, 32, 44, 48, 65, 45, 50,
                              49, 55, 54, 49, 43, 44, 43, 40, 46, 32, 44, 48};

    linearLayerFullPass(layer, X, Y, expected_Y, expected_dLdW, expected_dLdX);

    destroyTensor(X);
    destroyTensor(Y);
}

static void linearLayerFullPassSmallerBatch(Linear* layer) {
    size_t samples = SAMPLES / 2;

    Tensor* X = createCommonInput(1, samples, layer->in_features);

    Tensor* Y = createMatrix(samples, layer->out_features);

    /* Shape: samples * flat_features_out */
    double expected_Y[] = {
        449, 399, 294, 483, 418, 431, 415, 413, 362,
        492, 503, 284, 580, 471, 479, 459, 470, 429,
    };

    /* Shape: same as layer->weight */
    double expected_dLdW[] = {
        5,  6,  13, 4,  7,  14, 12, 12, 17, 11, 15, 12, 18, 11, 13, 5,  6,
        13, 4,  7,  14, 12, 12, 17, 11, 15, 12, 18, 11, 13, 5,  6,  13, 4,
        7,  14, 12, 12, 17, 11, 15, 12, 18, 11, 13, 5,  6,  13, 4,  7,  14,
        12, 12, 17, 11, 15, 12, 18, 11, 13, 5,  6,  13, 4,  7,  14, 12, 12,
        17, 11, 15, 12, 18, 11, 13, 5,  6,  13, 4,  7,  14, 12, 12, 17, 11,
        15, 12, 18, 11, 13, 5,  6,  13, 4,  7,  14, 12, 12, 17, 11, 15, 12,
        18, 11, 13, 5,  6,  13, 4,  7,  14, 12, 12, 17, 11, 15, 12, 18, 11,
        13, 5,  6,  13, 4,  7,  14, 12, 12, 17, 11, 15, 12, 18, 11, 13,
    };

    /* Shape: same as input */
    double expected_dLdX[] = {
        65, 45, 50, 49, 55, 54, 49, 43, 44, 43, 40, 46, 32, 44, 48,
        65, 45, 50, 49, 55, 54, 49, 43, 44, 43, 40, 46, 32, 44, 48,
    };

    linearLayerFullPass(layer, X, Y, expected_Y, expected_dLdW, expected_dLdX);

    destroyTensor(X);
    destroyTensor(Y);
}

static void linearLayerFullPassBiggerBatch(Linear* layer) {
    size_t samples = SAMPLES * 2;

    Tensor* X = createCommonInput(1, samples, layer->in_features);
    size_t already_filled = X->flat_size / 2;
    for (size_t i = 0; i < already_filled; i++) {
        X->data[i + already_filled] = input_tensor_array[i];
    }

    Tensor* Y = createMatrix(samples, layer->out_features);

    /* Shape: samples * flat_features_out */
    double expected_Y[] = {
        449, 399, 294, 483, 418, 431, 415, 413, 362, 492, 503, 284,
        580, 471, 479, 459, 470, 429, 329, 387, 271, 410, 372, 372,
        390, 327, 348, 335, 356, 259, 348, 410, 336, 365, 300, 285,
        449, 399, 294, 483, 418, 431, 415, 413, 362, 492, 503, 284,
        580, 471, 479, 459, 470, 429, 329, 387, 271, 410, 372, 372,
        390, 327, 348, 335, 356, 259, 348, 410, 336, 365, 300, 285};

    /* Shape: same as layer->weight */
    double expected_dLdW[] = {
        34, 36, 48, 30, 34, 38, 38, 42, 44, 48, 36, 34, 58, 46, 32, 34, 36,
        48, 30, 34, 38, 38, 42, 44, 48, 36, 34, 58, 46, 32, 34, 36, 48, 30,
        34, 38, 38, 42, 44, 48, 36, 34, 58, 46, 32, 34, 36, 48, 30, 34, 38,
        38, 42, 44, 48, 36, 34, 58, 46, 32, 34, 36, 48, 30, 34, 38, 38, 42,
        44, 48, 36, 34, 58, 46, 32, 34, 36, 48, 30, 34, 38, 38, 42, 44, 48,
        36, 34, 58, 46, 32, 34, 36, 48, 30, 34, 38, 38, 42, 44, 48, 36, 34,
        58, 46, 32, 34, 36, 48, 30, 34, 38, 38, 42, 44, 48, 36, 34, 58, 46,
        32, 34, 36, 48, 30, 34, 38, 38, 42, 44, 48, 36, 34, 58, 46, 32};

    /* Shape: same as input */
    double expected_dLdX[] = {
        65, 45, 50, 49, 55, 54, 49, 43, 44, 43, 40, 46, 32, 44, 48, 65, 45, 50,
        49, 55, 54, 49, 43, 44, 43, 40, 46, 32, 44, 48, 65, 45, 50, 49, 55, 54,
        49, 43, 44, 43, 40, 46, 32, 44, 48, 65, 45, 50, 49, 55, 54, 49, 43, 44,
        43, 40, 46, 32, 44, 48, 65, 45, 50, 49, 55, 54, 49, 43, 44, 43, 40, 46,
        32, 44, 48, 65, 45, 50, 49, 55, 54, 49, 43, 44, 43, 40, 46, 32, 44, 48,
        65, 45, 50, 49, 55, 54, 49, 43, 44, 43, 40, 46, 32, 44, 48, 65, 45, 50,
        49, 55, 54, 49, 43, 44, 43, 40, 46, 32, 44, 48};

    linearLayerFullPass(layer, X, Y, expected_Y, expected_dLdW, expected_dLdX);

    destroyTensor(X);
    destroyTensor(Y);
}

void test_linearForward(void) {
    Tensor* X = createCommonInput(SAMPLES, ROWS, IN_FEATURES);

    Linear* layer = createLinearLayer(IN_FEATURES, OUT_FEATURES);

    /* bias vector of 1 */
    destroyTensor(layer->bias->param);
    layer->bias->param = onesVector(OUT_FEATURES);

    /* initialize output */
    Tensor* Y = createTensor(SAMPLES, ROWS, OUT_FEATURES);

    /* adjust the expected output to account for the bias vector */
    double expected[SAMPLES * ROWS * OUT_FEATURES] = {
        18, 36, 54,  31, 62, 93,  30, 60, 90,  17, 34, 51,
        35, 70, 105, 39, 78, 117, 34, 68, 102, 21, 42, 63,
        11, 22, 33,  22, 44, 66,  18, 36, 54,  23, 46, 69};
    for (size_t i = 0; i < Y->flat_size; i++) {
        expected[i] += layer->bias->param->data[0];
    }

    /* initialize the weights */
    for (size_t i = 1; i < OUT_FEATURES + 1; i++) {
        for (size_t j = 0; j < IN_FEATURES; j++) {
            size_t idx = indexMatrix(layer->weights->param, i - 1, j);
            layer->weights->param->data[idx] = (double)i;
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

void test_linearBackward3d(void) {
    Tensor* X = createCommonInput(SAMPLES, ROWS, IN_FEATURES);

    Tensor* Y = createTensor(SAMPLES, ROWS, OUT_FEATURES);

    Linear* layer = createLinearLayer(IN_FEATURES, OUT_FEATURES);
    /* No need to randomly initialize the weights, they are controlled in
these
     * tests */

    /* To check the correctness of the math, we control the learnable params
     * default initialization */
    /* Below weights are transposed (same happens inside the linear layer).
     */
    /* Shape: features_in * features_out */
    double weights_T[OUT_FEATURES * IN_FEATURES] = {2, 3, 2, 4, 2, 3, 4, 3,
                                                    6, 9, 2, 8, 1, 1, 1};
    for (size_t i = 0; i < layer->weights->param->flat_size; i++) {
        layer->weights->param->data[i] = weights_T[i];
    }

    /* Shape: features_out */
    double bias[] = {2, 5, 8};
    for (size_t i = 0; i < layer->bias->param->flat_size; i++) {
        layer->bias->param->data[i] = bias[i];
    }

    double expected_Y[SAMPLES * ROWS * OUT_FEATURES] = {
        49, 85,  64, 88,  154, 103, 74, 133, 101, 39, 90,  34,
        94, 183, 77, 102, 207, 83,  91, 189, 96,  50, 112, 46,
        31, 54,  28, 59,  94,  72,  49, 115, 63,  70, 117, 60};

    double expected_dLdW[OUT_FEATURES * IN_FEATURES] = {
        54, 54, 74, 60, 57, 54, 54, 74, 60, 57, 54, 54, 74, 60, 57};

    double expected_dLdX[SAMPLES * ROWS * IN_FEATURES] = {
        7, 15, 6, 11, 12, 7, 15, 6, 11, 12, 7, 15, 6, 11, 12, 7, 15, 6, 11, 12,
        7, 15, 6, 11, 12, 7, 15, 6, 11, 12, 7, 15, 6, 11, 12, 7, 15, 6, 11, 12,
        7, 15, 6, 11, 12, 7, 15, 6, 11, 12, 7, 15, 6, 11, 12, 7, 15, 6, 11, 12};

    linearLayerFullPass(layer, X, Y, expected_Y, expected_dLdW, expected_dLdX);

    destroyTensor(X);
    destroyTensor(Y);
    destroyLinearLayer(layer);
}

void test_linearBackward(void) {
    Linear* layer = createCommonLinearLayer();

    linearLayerFullPassNormalBatch(layer);

    destroyLinearLayer(layer);
}

void test_differentBatchSizeDuringTraining(void) {
    Linear* layer = createCommonLinearLayer();

    linearLayerFullPassNormalBatch(layer);

    linearLayerFullPassSmallerBatch(layer);

    linearLayerFullPassBiggerBatch(layer);

    destroyLinearLayer(layer);
}

void test_finiteDiffLinearGradients(void) {
    gsl_rng* rng = initRNG();
    size_t iters = 5;

    for (size_t i = 0; i < iters; i++) {
        /* Get some random shapes, excluding 0 */
        size_t n = gsl_rng_uniform_int(rng, 20) + 1;
        size_t m = gsl_rng_uniform_int(rng, 100) + 1;
        size_t k = gsl_rng_uniform_int(rng, 10) + 1;

        /* Test one iteration with 3D inputs too */
        bool is_last_iter = i == iters - 1;
        size_t batch = is_last_iter ? gsl_rng_uniform_int(rng, 10) + 1 : 1;

        printf("Shape: (%lu, %lu, %lu) -> (%lu, %lu, %lu)\n", batch, n, m,
               batch, n, k);

        Linear* layer = createLinearLayer(m, k);
        layer->init(layer, rng);

        Tensor* X = createTensor(batch, n, m);
        randUniformTensor(X, 1, rng);

        /* Step size for finite differences */
        double epsilon = 1e-5;
        /* Max allowed relative error */
        double tolerance = 1e-4;

        double err_w = finiteDiffLinearWeights(layer, X, epsilon, rng);
        printf("Max Relative Error (Weights): %e\n", err_w);
        TEST_ASSERT_DOUBLE_WITHIN(tolerance, 0.0, err_w);

        double err_b = finiteDiffLinearBias(layer, X, epsilon, rng);
        printf("Max Relative Error (Bias):    %e\n", err_b);
        TEST_ASSERT_DOUBLE_WITHIN(tolerance, 0.0, err_b);

        double err_x = finiteDiffLinearInput(layer, X, epsilon, rng);
        printf("Max Relative Error (Input):   %e\n", err_x);
        TEST_ASSERT_DOUBLE_WITHIN(tolerance, 0.0, err_x);

        destroyTensor(X);
        destroyLinearLayer(layer);
        printf("\n");
    }

    destroyRNG(rng);
}

#endif  // TEST
