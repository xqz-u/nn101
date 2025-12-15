#ifdef TEST

#include "unity.h"

#include <gsl/gsl_statistics_double.h>
#include <string.h>

#include "activation.h"
#include "learnable_param.h"
#include "linear.h"
#include "loss.h"
#include "mlp.h"
#include "nn_math.h"
#include "sgd.h"
#include "tensor.h"
#include "utils.h"

void setUp(void) {}

void tearDown(void) {}

void mlpXOR(size_t dimensions[],
            size_t n_dimensions,
            size_t epochs,
            double lr) {
    gsl_rng* rng = initRNG();

    /* 1. Input: XOR truth table (0,0), (0,1), (1,0), (1,1) */
    size_t input_features = 2;
    size_t batch = 4;

    Tensor* X = createTensor(batch, 1, input_features);
    double x_vals[] = {0, 0, 0, 1, 1, 0, 1, 1};
    memcpy(X->data, x_vals, batch * input_features * sizeof(double));

    size_t classes = 2;
    size_t labels[] = {0, 1, 1, 0};

    /* 2. Model */
    MLP model = createMLP(n_dimensions, dimensions, rng);

    /* 3. Training components */

    /* weights matrix + bias vector for each block */
    size_t tot_learnable_params = model.n_layers * 2;
    LearnableParam* params[tot_learnable_params];

    size_t p_idx = 0;
    for (size_t i = 0; i < model.n_layers; i++) {
        Linear* layer = model.linears[i];
        params[p_idx++] = layer->weights;
        params[p_idx++] = layer->bias;
    }

    SGD optim = createSGD(lr, params, tot_learnable_params);

    CrossEntropyLoss loss_fn = createCrossEntropyLoss(classes);

    /* 4. Buffers */
    Tensor* loss_val = createVector(batch);
    Tensor* X_grad = zerosLike(X);

    /* 5. Training loop */
    double errors_track[epochs];
    double prev_err = INFINITY;

    size_t output_idx = model.n_layers - 1;

    for (size_t i = 0; i < epochs; i++) {
        optim.zeroGrads(&optim);

        model.forward(&model, X);

        loss_fn.forward(&loss_fn, model.logits[output_idx], labels, loss_val);
        double new_err = gsl_stats_mean(loss_val->data, 1, loss_val->flat_size);
        TEST_ASSERT_LESS_OR_EQUAL_DOUBLE(prev_err, new_err);

        prev_err = new_err;
        errors_track[i] = new_err;

        /* TODO ensure the capacity of logits & activations in one pass, do not
         * differentiate between .param & .grad; do it both in forward &
         * backward pass to be safe. */
        ensureTensorCapacity(&model.logits_grad[output_idx],
                             model.logits[output_idx]);

        loss_fn.backward(&loss_fn, labels, model.logits_grad[output_idx]);

        model.backward(&model, X_grad);

        optim.step(&optim);
    }

    printf("Training errors:\n");
    printRealArray(errors_track, epochs, "%.5f");

    model.forward(&model, X);

    size_t decisions[batch];
    simpleInference(batch, classes, model.logits[output_idx], decisions);
    printf("Predictions:\n");
    printIntArray(decisions, batch);
    TEST_ASSERT_EQUAL_INT_ARRAY(labels, decisions, batch);

    printf("Accuracy: %.2f\n", accuracyScore(decisions, labels, batch));

    destroyMLP(&model);
    destroySGD(&optim);
    destroyCrossEntropyLoss(&loss_fn);
    destroyTensors(3, X, X_grad, loss_val);
    destroyRNG(rng);
}

void test_smallMLP(void) {
    size_t dimensions[] = {2, 8, 2};
    size_t n_dimensions = sizeof(dimensions) / sizeof(dimensions[0]);
    mlpXOR(dimensions, n_dimensions, 275, 1e-1);
}

void test_deepMLP(void) {
    size_t dimensions[] = {2, 8, 2};
    size_t n_dimensions = sizeof(dimensions) / sizeof(dimensions[0]);
    mlpXOR(dimensions, n_dimensions, 150, 1e-1);
}

#endif  // TEST
