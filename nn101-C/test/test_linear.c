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
    Tensor* X = createTensor(SAMPLES, ROWS, IN_FEATURES);
    for (size_t i = 0; i < X->flat_size; i++) {
        X->data[i] = input_tensor_array[i];
    }

    Tensor* Y = createTensor(SAMPLES, ROWS, OUT_FEATURES);

    Linear* layer = createLinearLayer(IN_FEATURES, OUT_FEATURES);

    /* To check the correctness of the math, we control the learnable params
     * default initialization */
    layer->setup(layer, X, rng);

    /* Weights are stored in their transposed form inside the linear layer */
    double weights_T[OUT_FEATURES * IN_FEATURES] = {8, 3, 2, 8, 6,
                                                    /*  */
                                                    9, 4, 9, 4, 0,
                                                    /*  */
                                                    3, 1, 0, 2, 5};
    for (size_t i = 0; i < layer->weights->flat_size; i++) {
        layer->weights->data[i] = weights_T[i];
    }

    double bias[OUT_FEATURES] = {8, 7, 3};
    for (size_t i = 0; i < layer->bias->flat_size; i++) {
        layer->bias->data[i] = bias[i];
    }

    layer->forward(layer, X, Y);

    double expected_Y[SAMPLES * ROWS * OUT_FEATURES] = {93, 111, 33,
                                                        //
                                                        190, 170, 71,
                                                        //
                                                        146, 191, 56,
                                                        /*  */
                                                        81, 105, 37,
                                                        //
                                                        200, 194, 78,
                                                        //
                                                        228, 212, 94,
                                                        /*  */
                                                        191, 162, 84,
                                                        //
                                                        106, 122, 48,
                                                        //
                                                        65, 77, 21,
                                                        /*  */
                                                        127, 151, 42,
                                                        //
                                                        107, 66, 55,
                                                        //
                                                        126, 131, 38};
    TEST_ASSERT_EQUAL_DOUBLE_ARRAY(expected_Y, Y->data, Y->flat_size);

    /* Imagine that the loss is just the sum of the output; then, the
    gradient
     * wrt to the loss is just a 1s tensor. */
    Tensor* dLdY = onesLike(Y);
    Tensor* dLdX = createTensor(SAMPLES, ROWS, IN_FEATURES);

    /* layer->backward(layer, dLdY, dLdX); */

    /* double expected_dLdW[IN_FEATURES * OUT_FEATURES] = {34, 38, 42, */
    /*                                                     /\*  *\/ */
    /*                                                     46, 50, 34, */
    /*                                                     /\*  *\/ */
    /*                                                     38, 42, 46, */
    /*                                                     /\*  *\/ */
    /*                                                     50, 34, 38, */
    /*                                                     /\*  *\/ */
    /*                                                     42, 46, 50}; */
    /* printMatrix(layer->weights); */
    /* printMatrix(layer->weights_grad); */
    /* printMatrix(layer->bias); */
    /* printMatrix(layer->bias_grad); */
    /* TEST_ASSERT_EQUAL_DOUBLE_ARRAY(expected_dLdW, layer->weights_grad->data,
     */
    /*                                layer->weights_grad->flat_size); */

    /* double expected_dLdb[OUT_FEATURES] = {4, 4, 4}; */
    /* TEST_ASSERT_EQUAL_DOUBLE_ARRAY(expected_dLdb, layer->bias_grad->data, */
    /*                                layer->bias_grad->flat_size); */

    /* double expected_dLdX[ROWS * IN_FEATURES] = {13, 13, 18, 3, 17, */
    /*                                             /\*  *\/ */
    /*                                             13, 13, 18, 3, 17, */
    /*                                             /\*  *\/ */
    /*                                             13, 13, 18, 3, 17, */
    /*                                             /\*  *\/ */
    /*                                             13, 13, 18, 3, 17}; */
    /* TEST_ASSERT_EQUAL_DOUBLE_ARRAY(expected_dLdX, dLdX->data,
     * dLdX->flat_size); */

    destroyTensor(dLdX);
    destroyTensor(dLdY);
    destroyTensor(X);
    destroyTensor(Y);
    destroyLinearLayer(layer);
}

#endif  // TEST
