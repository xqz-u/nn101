#include <math.h>

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

double computeMean(double array[], size_t size) {
    double sum = 0.f;
    for (size_t i = 0; i < size; i++) {
        sum += array[i];
    }
    return sum / size;
}

double computeStd(double array[], size_t size) {
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

void zeroGrads(LearnableParam* params[], size_t params_count) {
    for (size_t i = 0; i < params_count; i++) {
        tensorFill(params[i]->grad, 0);
    }
}

void printIntArray(size_t a[], size_t size) {
    if (a == nullptr) {
        return;
    }
    printf("[%lu", a[0]);
    for (size_t i = 1; i < size; i++) {
        printf(", %lu", a[i]);
    }
    printf("]\n");
}

void printRealArrayBase(double a[], size_t size, const char* fmt) {
    if (a == nullptr) {
        return;
    }
    printf("[");
    for (size_t i = 0; i < size; i++) {
        printf(fmt, a[i]);
        if (i != size - 1) {
            printf(", ");
        }
    }
    printf("]\n");
}

void var_printRealArray(PrintArgs args) {
    const char* final_fmt = args.fmt ? args.fmt : "%.2f";
    printRealArrayBase(args.array, args.size, final_fmt);
}

double max(double* a, size_t size) {
    double max = a[0];
    for (size_t i = 1; i < size; i++) {
        if (a[i] > max) {
            max = a[i];
        }
    }
    return max;
}

void simpleInference(size_t samples_count,
                     size_t classes,
                     Tensor* logits,
                     size_t predictions[samples_count]) {
    for (size_t i = 0; i < samples_count; i++) {
        Tensor tmp = {.flat_size = classes, .data = &logits->data[i * classes]};
        predictions[i] = argmax(&tmp);
    }
}
