#include <math.h>

#include "utils.h"

gsl_rng* initRNG(void) {
    const gsl_rng_type* T = NULL;
    gsl_rng* rng = NULL;

    gsl_rng_env_setup();

    T = gsl_rng_default;
    rng = gsl_rng_alloc(T);

    /* printf("RNG: '%s' seed: %lu\n", gsl_rng_name(rng), gsl_rng_default_seed);
     */

    return rng;
}

double computeMean(double* array, size_t size) {
    double sum = 0.f;
    for (size_t i = 0; i < size; i++) {
        sum += array[i];
    }
    return sum / size;
}

double computeStd(double* array, size_t size) {
    double avg = computeMean(array, size);

    double diffs = 0.f;
    for (size_t i = 0; i < size; i++) {
        double delta = array[i] - avg;
        diffs += delta * delta;
    }

    return sqrt(diffs / size);
}
