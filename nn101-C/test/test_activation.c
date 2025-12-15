#ifdef TEST

#include "unity.h"

#include <gsl/gsl_statistics_double.h>
#include <math.h>
#include <string.h>

#include "activation.h"
#include "learnable_param.h"
#include "linear.h"
#include "loss.h"
#include "nn_math.h"
#include "sgd.h"
#include "tensor.h"
#include "utils.h"

void setUp(void) {}

void tearDown(void) {}

void test_eluForward(void) {
    size_t batch = 3;
    Tensor* input = createTensor(batch, 1, 1);
    Tensor* output = createTensor(batch, 1, 1);

    input->data[0] = 10.0;
    input->data[1] = 0.0;
    input->data[2] = -1.0;

    double alpha = 1.0;
    ELU elu = createELUActivation(alpha);

    elu.forward(&elu, input, output);

    double expected_neg = alpha * (exp(-1.0) - 1.0);  // approx -0.632

    TEST_ASSERT_EQUAL_DOUBLE(10.0, output->data[0]);
    TEST_ASSERT_EQUAL_DOUBLE(0.0, output->data[1]);
    TEST_ASSERT_DOUBLE_WITHIN(1e-6, expected_neg, output->data[2]);

    destroyTensors(2, input, output);
}

void test_eluBackward(void) {
    size_t batch = 3;
    /* y */
    Tensor* activations = createTensor(batch, 1, 1);
    /* Incoming gradient (dL/dy), set to 1 for simplicity */
    Tensor* grad_out = onesTensor(batch, 1, 1);
    /* Output gradient, wrt to input (dL/dx) */
    Tensor* grad_in = createTensor(batch, 1, 1);

    double alpha = 1.0;
    ELU elu = createELUActivation(alpha);

    /* Positive activation, derivative should be constant 1 */
    activations->data[0] = 5.0;
    /* Zero activation, derivative should be alpha * grad_in->data[1] = 1 */
    activations->data[1] = 0.0;
    /* Negative activation, corresponding to x=-1 */
    double y_neg = alpha * (exp(-1.0) - 1.0);
    activations->data[2] = y_neg;

    elu.backward(&elu, activations, grad_out, grad_in);

    double expected_grad_neg = y_neg + alpha;
    double expected_grad_neg_no_trick = alpha * exp(-1.0);

    TEST_ASSERT_EQUAL_DOUBLE(1.0, grad_in->data[0]);
    TEST_ASSERT_EQUAL_DOUBLE(1.0, grad_in->data[1]);
    TEST_ASSERT_EQUAL_DOUBLE(expected_grad_neg_no_trick, expected_grad_neg);

    destroyTensors(3, activations, grad_out, grad_in);
}

void test_eluXOR(void) {
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
    size_t hidden_features = 8;  // The higher, the faster the convergence
    Linear* fc1 = createLinearLayer(input_features, hidden_features);
    Linear* fc2 = createLinearLayer(hidden_features, classes);
    fc1->init(fc1, rng);
    fc2->init(fc2, rng);

    ELU elu = createELUActivation(1.0);

    /* 3. Training components */
    LearnableParam* params[] = {fc1->weights, fc1->bias, fc2->weights,
                                fc2->bias};
    double lr = 1e-1;
    SGD optim = createSGD(lr, params, 4);

    CrossEntropyLoss loss_fn = createCrossEntropyLoss(classes);

    /* 4. Buffers */
    Tensor* h1 = createTensor(batch, 1, hidden_features);  // logits fc1
    Tensor* h1_act = createTensor(batch, 1, hidden_features);
    Tensor* logits = createTensor(batch, 1, classes);
    Tensor* loss_val = createVector(batch);

    /* Gradients */
    Tensor* d_logits = zerosLike(logits);
    Tensor* d_h1_act = zerosLike(h1_act);
    Tensor* d_h1 = zerosLike(h1);
    Tensor* d_X = zerosLike(X);

    /* 5. Training loop */
    size_t epochs = 275;
    double errors_track[epochs];
    double prev_err = INFINITY;

    for (size_t i = 0; i < epochs; i++) {
        optim.zeroGrads(&optim);

        /* Forward */
        fc1->forward(fc1, X, h1);
        elu.forward(&elu, h1, h1_act);
        fc2->forward(fc2, h1_act, logits);

        /* Loss */
        loss_fn.forward(&loss_fn, logits, labels, loss_val);

        double new_err = gsl_stats_mean(loss_val->data, 1, loss_val->flat_size);
        TEST_ASSERT_LESS_OR_EQUAL_DOUBLE(prev_err, new_err);

        prev_err = new_err;
        errors_track[i] = new_err;

        /* Backprop */
        loss_fn.backward(&loss_fn, labels, d_logits);

        fc2->backward(fc2, d_logits, d_h1_act);
        elu.backward(&elu, h1_act, d_h1_act, d_h1);
        fc1->backward(fc1, d_h1, d_X);

        optim.step(&optim);
    }

    printf("Training errors:\n");
    printRealArray(errors_track, epochs, "%.5f");

    /* 6. Verification */
    fc1->forward(fc1, X, h1);
    elu.forward(&elu, h1, h1_act);
    fc2->forward(fc2, h1_act, logits);

    size_t decisions[batch];
    simpleInference(batch, classes, logits, decisions);
    printf("Predictions:\n");
    printIntArray(decisions, batch);
    TEST_ASSERT_EQUAL_INT_ARRAY(labels, decisions, batch);

    printf("Accuracy: %.2f\n", accuracyScore(decisions, labels, batch));

    destroyLinearLayer(fc1);
    destroyLinearLayer(fc2);
    destroySGD(&optim);
    destroyCrossEntropyLoss(&loss_fn);
    destroyTensors(9, X, h1, h1_act, logits, loss_val, d_logits, d_h1_act, d_h1,
                   d_X);
    destroyRNG(rng);
}

#endif  // TEST
