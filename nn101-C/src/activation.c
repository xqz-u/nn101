#include <assert.h>
#include <math.h>

#include "activation.h"

double eluScalar(double x, double alpha) {
    return x > 0 ? x : alpha * (exp(x) - 1);
}

/* Wehen x < 0, the derivative is y' = alpha * e^x. Then ELU where x < 0 can be
 * rewritten as alpha * e^x = y + alpha; so in this case we use y' = y + alpha &
 * avoid computing e^x again. Thanks Gemini! */
double eluScalarGrad(double activation, double alpha) {
    return activation > 0 ? 1 : activation + alpha;
}

void eluForward(ELU* self, Tensor* logits, Tensor* activations) {
    assert(logits->batch == activations->batch && logits->n == activations->n &&
           logits->features == activations->features);

    for (size_t i = 0; i < logits->flat_size; i++) {
        activations->data[i] = eluScalar(logits->data[i], self->alpha);
    }
}

void eluBackward(ELU* self,
                 Tensor* activations,
                 Tensor* grad_out,
                 Tensor* grad_in) {
    assert(grad_in->batch == grad_out->batch && grad_in->n == grad_out->n &&
           grad_in->features == grad_out->features);

    for (size_t i = 0; i < grad_out->flat_size; i++) {
        grad_in->data[i] = grad_out->data[i] *
                           eluScalarGrad(activations->data[i], self->alpha);
    }
}

ELU createELUActivation(double alpha) {
    return (ELU){
        .alpha = alpha, .forward = &eluForward, .backward = &eluBackward};
}
