#ifdef TEST

#include "unity.h"

#include <gsl/gsl_rng.h>
#include <unistd.h>

#include "linear.h"
#include "nn_math.h"
#include "tensor.h"
#include "utils.h"

#define SAMPLES 4
#define IN_COUNT 3
#define IN_FEATURES 5
#define OUT_FEATURES 2
#define INPUT_FLAT_SIZE SAMPLES * IN_COUNT * IN_FEATURES
#define OUTPUT_FLAT_SIZE SAMPLES * IN_COUNT * OUT_FEATURES

static const gsl_rng* rng = NULL;

static const double input_tensor_array[INPUT_FLAT_SIZE] = {3, 5, 5, 3, 2,
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
    double expected[OUTPUT_FLAT_SIZE] = {18, 36,
                                         //
                                         31, 62,
                                         //
                                         30, 60,
                                         /*  */
                                         17, 34,
                                         //
                                         35, 70,
                                         //
                                         39, 78,
                                         /*  */
                                         34, 68,
                                         //
                                         21, 42,
                                         //
                                         11, 22,
                                         /*  */
                                         22, 44,
                                         //
                                         18, 36,
                                         //
                                         23, 46};

    /* initialize input */
    Tensor* X = createTensor(SAMPLES, IN_COUNT, IN_FEATURES);
    for (size_t i = 0; i < INPUT_FLAT_SIZE; i++) {
        X->data[i] = input_tensor_array[i];
    }
    /* printTensor(X); */

    Linear* layer = createLinearLayer(IN_FEATURES, OUT_FEATURES);

    /* bias vector of 1 */
    for (size_t i = 0; i < layer->bias->features; i++) {
        layer->bias->data[i] = 1.f;
    }
    /* printVector(layer->bias); */

    /* adjust the expected output to account for the bias vector */
    for (size_t i = 0; i < OUTPUT_FLAT_SIZE; i++) {
        expected[i] += layer->bias->data[0];
    }

    /* initialize the weights */
    for (size_t i = 1; i < OUT_FEATURES + 1; i++) {
        for (size_t j = 0; j < IN_FEATURES; j++) {
            size_t idx = indexMatrix(layer->weights, i - 1, j);
            layer->weights->data[idx] = (double)i;
        }
    }
    /* printMatrix(layer->weights); */

    /* initialize output */
    Tensor* Y = createTensor(SAMPLES, IN_COUNT, OUT_FEATURES);

    layer->forward(layer, X, Y);
    /* printTensor(Y); */

    for (size_t i = 0; i < OUTPUT_FLAT_SIZE; i++) {
        TEST_ASSERT_EQUAL_DOUBLE(Y->data[i], expected[i]);
    }

    destroyTensor(X);
    destroyTensor(Y);
    destroyLinearLayer(layer);
}

/* void test_linearBackward(void) { */
/*     Linear* layer = createLinearLayer(6, 3); */

/*     Tensor* input = createTensor(1, 4, 6); */
/*     for (size_t i = 0; i < input->flat_size; i++) { */
/*         input[i] = (double)i + 1; */
/*     } */
/*     Tensor* output = createTensor(1, 4, 3); */

/*     layer->setup(layer, input, rng); */
/*     layer->forward(layer, input, output); */
/* } */

#endif  // TEST
