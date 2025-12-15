#include "mlp.h"

void forwardMLP(MLP* model, Tensor* X) {
    for (size_t i = 0; i < model->n_layers; i++) {
        if (i != 0) {
            size_t act_idx = i - 1;
            ensureTensorCapacity(&model->activations[act_idx], X);
            model->activation.forward(&model->activation, X,
                                      model->activations[act_idx]);
            X = model->activations[act_idx];
        }

        size_t out_features = model->linears[i]->out_features;
        Tensor Y_stub = (Tensor){.batch = X->batch,
                                 .n = X->n,
                                 .features = out_features,
                                 .flat_size = X->batch * X->n * out_features};

        ensureTensorCapacity(&model->logits[i], &Y_stub);

        model->linears[i]->forward(model->linears[i], X, model->logits[i]);
        X = model->logits[i];
    }
}

void backwardMLP(MLP* model, Tensor* X_grad) {
    Tensor* grad_out = model->logits_grad[model->n_layers - 1];
    ELU h = model->activation;

    /* Mixing decrement (post-order) & guard evaluation ensures no wrap-around
     * of unsigned ints below 0 -> no infinite loop. */
    for (size_t i = model->n_layers; i-- > 1;) {
        Linear* linear = model->linears[i];
        size_t prev_layer_idx = i - 1;

        size_t in_features = model->linears[i]->in_features;
        Tensor Y_stub =
            (Tensor){.batch = grad_out->batch,
                     .n = grad_out->n,
                     .features = in_features,
                     .flat_size = grad_out->batch * grad_out->n * in_features};
        ensureTensorCapacity(&model->activations_grad[prev_layer_idx], &Y_stub);
        Tensor* linear_input_grad = model->activations_grad[prev_layer_idx];
        linear->backward(linear, grad_out, linear_input_grad);

        ensureTensorCapacity(&model->logits_grad[prev_layer_idx],
                             linear_input_grad);
        Tensor* h_input_grad = model->logits_grad[prev_layer_idx];
        h.backward(&h, model->activations[prev_layer_idx], linear_input_grad,
                   h_input_grad);

        grad_out = h_input_grad;
    }

    Linear* input_layer = model->linears[0];
    input_layer->backward(input_layer, grad_out, X_grad);
}

MLP createMLP(size_t n_features, size_t features[], gsl_rng* rng) {
    size_t n_layers = n_features - 1;

    Tensor** logits = calloc(n_layers, sizeof(Tensor*));
    Tensor** logits_grad = calloc(n_layers, sizeof(Tensor*));

    Tensor** activations = calloc(n_layers - 1, sizeof(Tensor*));
    Tensor** activations_grad = calloc(n_layers - 1, sizeof(Tensor*));

    Linear** linears = malloc(n_layers * sizeof(Linear*));

    for (size_t i = 0; i < n_layers; i++) {
        size_t in_features = features[i];
        size_t out_features = features[i + 1];

        Linear* layer = createLinearLayer(in_features, out_features);
        layer->init(layer, rng);
        linears[i] = layer;
    }

    /* NOTE `features` is shallow-copied & will be garbage once the original
     * stack object goes out of scope. */
    return (MLP){.n_features = n_features,
                 .features = features,
                 .n_layers = n_layers,
                 .activation = createELUActivation(1.0),
                 .linears = linears,

                 .logits = logits,
                 .logits_grad = logits_grad,
                 .activations = activations,
                 .activations_grad = activations_grad,

                 .forward = &forwardMLP,
                 .backward = &backwardMLP};
}

void destroyMLP(MLP* model) {
    for (size_t i = 0; i < model->n_layers; i++) {
        destroyLinearLayer(model->linears[i]);

        destroyTensor(model->logits[i]);
        destroyTensor(model->logits_grad[i]);

        if (i < model->n_layers - 1) {
            destroyTensor(model->activations[i]);
            destroyTensor(model->activations_grad[i]);
        }
    }
    free(model->linears);
    free(model->logits);
    free(model->logits_grad);
    free(model->activations);
    free(model->activations_grad);
}
