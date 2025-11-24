#include <assert.h>

#include "loss.h"

double mse_scalar(double y_prime, double y) {
    return (y_prime - y) * (y_prime - y);
}

double mse_grad_scalar(double pred, double target) {
    return 2 * (pred - target);
}

void mse_loss(Tensor* pred, Tensor* target, Tensor* out) {
    assert(pred->features == target->features &&
           pred->flat_size == target->flat_size &&
           target->flat_size == out->flat_size);

    for (size_t i = 0; i < out->flat_size; i++) {
        out->data[i] = mse_scalar(pred->data[i], target->data[i]);
    }
}

void mse_grad(Tensor* pred, Tensor* target, Tensor* out) {
    assert(pred->features == target->features &&
           pred->flat_size == target->flat_size &&
           target->flat_size == out->flat_size);

    for (size_t i = 0; i < out->flat_size; i++) {
        out->data[i] = mse_grad_scalar(pred->data[i], target->data[i]);
    }
}

MSELoss createMSE(void) {
    return (MSELoss){.forward = &mse_loss, .backward = &mse_grad};
}
