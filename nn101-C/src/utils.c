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
