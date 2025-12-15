#include <assert.h>
#include <math.h>

#include "loss.h"
#include "nn_math.h"

double mseScalar(double y_prime, double y) {
    return (y_prime - y) * (y_prime - y);
}

double mseScalarGrad(double pred, double target) {
    return 2 * (pred - target);
}

void mseLoss(Tensor* pred, Tensor* target, Tensor* out) {
    assert(pred->features == target->features &&
           pred->flat_size == target->flat_size &&
           target->flat_size == out->flat_size);

    for (size_t i = 0; i < out->flat_size; i++) {
        out->data[i] = mseScalar(pred->data[i], target->data[i]);
    }
}

void mseLossGrad(Tensor* pred, Tensor* target, Tensor* out) {
    assert(pred->features == target->features &&
           pred->flat_size == target->flat_size &&
           target->flat_size == out->flat_size);

    for (size_t i = 0; i < out->flat_size; i++) {
        out->data[i] = mseScalarGrad(pred->data[i], target->data[i]);
    }
}

MSELoss createMSELoss(void) {
    return (MSELoss){.forward = &mseLoss, .backward = &mseLossGrad};
}

/* NOTE CE loss for a target c is nll(softmax(logits))
 * ie -log(e^x_i / LSE), with LSE = sum(e^x_j)
 * which becomes -x_i + LSE using logarithm rules. Avoiding the
 * division in softmax prevents possible over/underflows. */
void crossEntropyLoss(CrossEntropyLoss* self,
                      Tensor* logits,
                      size_t* labels,
                      Tensor* loss) {
    assert(logits->features == self->n_classes);

    Tensor logits_2d = {};
    flattenFirstDims(logits, &logits_2d);
    /* Output gradient is the unreduced vector of loss for each sample. */
    assert(loss->batch == 1 && loss->n == 1 && loss->features == logits_2d.n);

    /* Here, we should make sure that the `probabilities` field has enough */
    /* capacity to deal with variable batch size. */
    ensureTensorCapacity(&self->probabilities, logits);

    for (size_t i = 0; i < logits_2d.n; i++) {
        size_t sample_start = i * self->n_classes;
        double* sample_logits = &logits->data[sample_start];

        size_t target_class = labels[i];
        double sample_lse = logSumExp(sample_logits, self->n_classes);
        loss->data[i] = sample_lse - sample_logits[target_class];

        /* Keep using the LSE trick to *cache the probabilities* obtained from
         * logits (used in the backward pass): e^x_i / e^LSE = e^(x_i - LSE). */
        double* sample_probs = &self->probabilities->data[sample_start];
        for (size_t j = 0; j < self->n_classes; j++) {
            sample_probs[j] = exp(sample_logits[j] - sample_lse);
        }
    }
}

/* Excellent reference:
 * https://robotchinwag.com/posts/crossentropy-loss-gradient/ */
void crossEntropyLossGrad(CrossEntropyLoss* self,
                          size_t labels[],
                          Tensor* grad_logits) {
    assert(grad_logits->features == self->n_classes);

    Tensor grad_logits_2d = {};
    flattenFirstDims(grad_logits, &grad_logits_2d);

    for (size_t i = 0; i < grad_logits_2d.n; i++) {
        size_t sample_start = i * self->n_classes;
        double* sample_probs = &self->probabilities->data[sample_start];
        double* sample_grads = &grad_logits->data[sample_start];

        size_t target_class = labels[i];

        for (size_t j = 0; j < self->n_classes; j++) {
            double predicted_class_prob = sample_probs[j];
            sample_grads[j] = predicted_class_prob - (j == target_class);
        }
    }
}

CrossEntropyLoss createCrossEntropyLoss(size_t classes) {
    return (CrossEntropyLoss){.n_classes = classes,
                              .probabilities = nullptr,
                              .forward = &crossEntropyLoss,
                              .backward = &crossEntropyLossGrad};
}

void destroyCrossEntropyLoss(CrossEntropyLoss* loss) {
    if (loss == nullptr || loss->probabilities == nullptr) {
        return;
    }
    destroyTensor(loss->probabilities);
}
