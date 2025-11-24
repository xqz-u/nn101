#include <math.h>
#include "nn_types.h"

#include "utils.h"

gsl_rng* initRNG(void) {
    const gsl_rng_type* T = nullptr;
    gsl_rng* rng = nullptr;

    gsl_rng_env_setup();

    T = gsl_rng_default;
    rng = gsl_rng_alloc(T);

    /* printf("RNG: '%s' seed: %lu\n", gsl_rng_name(rng), gsl_rng_default_seed);
     */

    return rng;
}

void destroyRNG(gsl_rng* rng) {
    if (rng != nullptr) {
        gsl_rng_free(rng);
    }
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

double relativeError(double a, double b) {
    double diff = fabs(a - b);
    double sum = fabs(a) + fabs(b);
    /* Avoid division by zero */
    return (sum > 1e-8) ? diff / sum : diff;
}

void zeroGrads(LearnableParam params[], size_t params_count) {
    for (size_t i = 0; i < params_count; i++) {
        tensorFill(params[i].grad, 0.0);
    }
}
