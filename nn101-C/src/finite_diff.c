#include "finite_diff.h"
#include "utils.h"

double finiteDiffLinearLayerStep(double* param_to_nudge,
                                 double original_value,
                                 Linear* layer,
                                 Tensor* X,
                                 Tensor* Y,
                                 Tensor* grad_out,
                                 double eps) {
    *param_to_nudge = original_value + eps;

    layer->forward(layer, X, Y);

    double nudged_loss = dotProduct(Y, grad_out);

    return nudged_loss;
}

double finiteDiffLinearLayer(Linear* layer,
                             Tensor* grad_in,
                             Tensor* analytical_grad,
                             Tensor* param_to_check,
                             Tensor* X,
                             double eps,
                             gsl_rng* rng) {
    Tensor* Y = createTensor(X->batch, X->n, layer->out_features);
    layer->forward(layer, X, Y);

    /* Dummy random dL/dY coming from next layer during backprop */
    Tensor* grad_out = zerosLike(Y);
    randUniformTensor(grad_out, 0.5, rng);

    layer->backward(layer, grad_out, grad_in);

    double max_rel_err = 0.0;

    for (size_t i = 0; i < param_to_check->flat_size; i++) {
        double* original_param_p = &param_to_check->data[i];
        double original_param = *original_param_p;

        /* Step right (+ɛ) */
        double loss_plus = finiteDiffLinearLayerStep(
            original_param_p, original_param, layer, X, Y, grad_out, eps);

        /* Step left (-ɛ) */
        double loss_minus = finiteDiffLinearLayerStep(
            original_param_p, original_param, layer, X, Y, grad_out, -eps);

        /* Compute slope numerically */
        double num_grad = (loss_plus - loss_minus) / (2.0 * eps);
        /* NOTE If `analytical_grad` points to a field of `layer` that is
         * modified during the forward pass, be sure to apply the checks
         * against a deep copy (forward pass is run inside this loop). */
        double ana_grad = analytical_grad->data[i];

        /* Save max relative error, which subsumes all others. */
        double rel_err = relativeError(num_grad, ana_grad);
        if (rel_err > max_rel_err) {
            max_rel_err = rel_err;
        }

        /* Reset param for next iteration */
        param_to_check->data[i] = original_param;
    }

    destroyTensor(Y);
    destroyTensor(grad_out);

    return max_rel_err;
}

double finiteDiffLinearWeights(Linear* layer,
                               Tensor* X,
                               double eps,
                               gsl_rng* rng) {
    Tensor* grad_in = zerosLike(X);
    double ret = finiteDiffLinearLayer(layer, grad_in, layer->weights_grad,
                                       layer->weights, X, eps, rng);
    destroyTensor(grad_in);
    return ret;
}

double finiteDiffLinearBias(Linear* layer,
                            Tensor* X,
                            double eps,
                            gsl_rng* rng) {
    Tensor* grad_in = zerosLike(X);
    double ret = finiteDiffLinearLayer(layer, grad_in, layer->bias_grad,
                                       layer->bias, X, eps, rng);
    destroyTensor(grad_in);
    return ret;
}

double finiteDiffLinearInput(Linear* layer,
                             Tensor* X,
                             double eps,
                             gsl_rng* rng) {
    Tensor* grad_in = zerosLike(X);
    double ret = finiteDiffLinearLayer(layer, grad_in, grad_in, X, X, eps, rng);
    destroyTensor(grad_in);
    return ret;
}
