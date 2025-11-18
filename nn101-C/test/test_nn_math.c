#ifdef TEST

#include "unity.h"

#include <gsl/gsl_rng.h>
#include <math.h>

#include "nn_math.h"
#include "tensor.h"
#include "utils.h"

void setUp(void) {}

void tearDown(void) {}

void test_kaimingWithinBounds(void) {
    Tensor* M = createMatrix(10, 5);

    gsl_rng* rng = initRNG();

    kaimingUniform(M, rng);

    double bound = 1 / sqrt(computeFanInFanOut(M).in);

    for (size_t i = 0; i < M->flat_size; i++) {
        TEST_ASSERT_GREATER_OR_EQUAL_DOUBLE(-bound, M->data[i]);
        TEST_ASSERT_LESS_OR_EQUAL_DOUBLE(bound, M->data[i]);
    }

    gsl_rng_free(rng);
    destroyTensor(M);
}

void test_kaimingMeanStd(void) {
    Tensor* M = createMatrix(20, 20);

    gsl_rng* rng = initRNG();

    kaimingUniform(M, rng);

    double bound = 1 / sqrt(computeFanInFanOut(M).in);

    /* https://en.wikipedia.org/wiki/Continuous_uniform_distribution */
    double expected_mean = (-bound + bound) / 2;
    double expected_std = sqrt(pow(bound - -bound, 2) / 12);

    double mean = computeMean(M->data, M->flat_size);
    double std = computeStd(M->data, M->flat_size);

    TEST_ASSERT_DOUBLE_WITHIN(1e-2, expected_mean, mean);
    TEST_ASSERT_DOUBLE_WITHIN(1e-2, expected_std, std);

    gsl_rng_free(rng);
    destroyTensor(M);
}

#endif  // TEST
