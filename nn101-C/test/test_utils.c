#ifdef TEST

#include "unity.h"

#include <gsl/gsl_rng.h>

#include "utils.h"

void setUp(void) {}

void tearDown(void) {}

void test_gslReproducibility(void) {
    gsl_rng* rng1 = initRNG();
    size_t value1 = gsl_rng_get(rng1);

    gsl_rng* rng2 = initRNG();
    size_t value2 = gsl_rng_get(rng2);

    TEST_ASSERT_EQUAL_UINT(value1, value2);

    destroyRNG(rng1);
    destroyRNG(rng2);
}

void test_gslRandomness(void) {
    gsl_rng* rng = initRNG();

    size_t values1[10] = {};
    size_t values2[10] = {};

    size_t values_size = sizeof(values1) / sizeof(values1[0]);

    for (size_t i = 0; i < values_size; i++) {
        values1[i] = gsl_rng_get(rng);
        values2[i] = gsl_rng_get(rng);
    }

    bool one_different = false;
    for (size_t i = 0; i < values_size; i++) {
        if (values1[i] != values2[i]) {
            one_different = true;
            break;
        }
    }

    TEST_ASSERT_TRUE_MESSAGE(
        one_different,
        "Randomness test failed: subsequent calls produce same serie.");

    destroyRNG(rng);
}

#endif  // TEST
