#ifdef TEST

#include "unity.h"

#include <gsl/gsl_randist.h>
#include <gsl/gsl_statistics_double.h>

#include "learnable_param.h"
#include "linear.h"
#include "loss.h"
#include "nn_math.h"
#include "sgd.h"
#include "tensor.h"
#include "utils.h"

void setUp(void) {}

void tearDown(void) {}

void test_linearRegression(void) {
    gsl_rng* rng = initRNG();

    size_t batch = 8;
    size_t n = 10;
    size_t features = 1;

    /* Center the inputs uniformly in [-1, 1]; input normalization would be
     * performed normally. */
    Tensor* X = createTensor(batch, n, features);
    randUniformTensor(X, 1, rng);

    /* Dummy prediction target: y = 2x + 1 */
    Tensor* Y = createTensor(batch, n, features);
    for (size_t i = 0; i < Y->flat_size; i++) {
        Y->data[i] = 2 * X->data[i] + 1;
    }

    Linear* layer = createLinearLayer(1, 1);
    layer->init(layer, rng);

    LearnableParam* params[] = {layer->weights, layer->bias};
    SGD optim = createSGD(0.01, params, 2);

    Tensor* Y_pred = zerosLike(Y);

    MSELoss mse = createMSELoss();
    Tensor* error = zerosLike(Y);
    Tensor* loss_grad = createTensor(X->batch, X->n, layer->out_features);
    Tensor* X_grad = zerosLike(X);

    double prev_err = INFINITY;

    size_t epochs = 10;
    for (size_t i = 0; i < epochs; i++) {
        optim.zeroGrads(&optim);

        layer->forward(layer, X, Y_pred);

        mse.forward(Y_pred, Y, error);

        double new_err = gsl_stats_mean(error->data, 1, error->flat_size);
        TEST_ASSERT_LESS_OR_EQUAL_DOUBLE(prev_err, new_err);

        prev_err = new_err;

        mse.backward(Y_pred, Y, loss_grad);

        layer->backward(layer, loss_grad, X_grad);

        optim.step(&optim);
    }

    printf("Final error: %.5f\n", prev_err);

    printf("True train Y:\n");
    printRealArray(Y->data, 10);
    printf("Predicted:\n");
    layer->forward(layer, X, Y_pred);
    printRealArray(Y_pred->data, 10);

    destroySGD(&optim);
    destroyLinearLayer(layer);
    destroyTensors(6, error, X_grad, loss_grad, X, Y, Y_pred);
    destroyRNG(rng);
}

void test_binaryClassification(void) {
    gsl_rng* rng = initRNG();

    size_t batch = 10;
    size_t n = 1;
    size_t features = 2;  // Cartesian coordinates

    size_t classes = 2;
    size_t labels[] = {0, 1, 1, 1, 0, 0, 1, 0, 1, 0};  // Size: batch

    Tensor* X = createTensor(batch, n, features);
    Tensor* X_grad = zerosLike(X);

    /* Samples belong to 2 clusters centered around (±2, ±2) */
    for (size_t i = 0; i < batch; i++) {
        double center = (labels[i] == 0) ? -2.0 : 2.0;
        /* Add some noise around the center */
        for (size_t j = 0; j < features; j++) {
            double noise = gsl_ran_flat(rng, -0.5, 0.5);
            X->data[i * n * features + j] = center + noise;
        }
    }

    printIntArray(labels, batch);
    printTensor(X);

    Linear* layer = createLinearLayer(features, classes);
    /* NOTE Not initializing the weights to make learning start from a random
     * guess */

    LearnableParam* params[] = {layer->weights, layer->bias};
    SGD optim = createSGD(0.1, params, 2);

    Tensor* logits = createTensor(batch, n, classes);
    Tensor* loss_grad = zerosLike(logits);

    CrossEntropyLoss ceLoss = createCrossEntropyLoss(classes);
    Tensor* loss = createVector(batch);

    size_t epochs = 5;
    double prev_err = INFINITY;
    double errors_track[epochs];
    /* To perform inference on the training set */
    size_t decisions[batch];

    for (size_t i = 0; i < epochs; i++) {
        optim.zeroGrads(&optim);

        layer->forward(layer, X, logits);

        ceLoss.forward(&ceLoss, logits, labels, loss);

        double new_err = gsl_stats_mean(loss->data, 1, loss->flat_size);
        TEST_ASSERT_LESS_OR_EQUAL_DOUBLE(prev_err, new_err);

        prev_err = new_err;
        errors_track[i] = new_err;

        simpleInference(batch, classes, logits, decisions);
        printf("Predictions:\n");
        printIntArray(decisions, batch);
        printf("Accuracy: %.2f\n", accuracyScore(decisions, labels, batch));

        ceLoss.backward(&ceLoss, labels, loss_grad);
        layer->backward(layer, loss_grad, X_grad);

        optim.step(&optim);
    }

    printf("Final error: %.5f, training errors:\n", prev_err);
    printRealArray(errors_track, epochs, "%.5f");

    layer->forward(layer, X, logits);
    simpleInference(batch, classes, logits, decisions);
    printf("Predictions:\n");
    printIntArray(decisions, batch);
    printf("Accuracy: %.2f\n", accuracyScore(decisions, labels, batch));

    destroyCrossEntropyLoss(&ceLoss);
    destroySGD(&optim);
    destroyLinearLayer(layer);
    destroyTensors(5, loss, loss_grad, X, X_grad, logits);
    destroyRNG(rng);
}

void test_multiClassClassification(void) {
    gsl_rng* rng = initRNG();

    size_t batch = 30;  // 10 samples per class
    size_t n = 1;
    size_t features = 2;

    size_t classes = 3;
    size_t labels[batch];

    Tensor* X = createTensor(batch, n, features);
    Tensor* X_grad = zerosLike(X);

    /* Dummy data: 3 blobs in a triangle */
    /* Class 0: (-2, -2) [Bottom Left] */
    /* Class 1: ( 2, -2) [Bottom Right] */
    /* Class 2: ( 0,  2) [Top Center] */
    /* classes x features */
    double centers[3][2] = {{-2.0, -2.0}, {2.0, -2.0}, {0.0, 2.0}};

    for (size_t i = 0; i < batch; i++) {
        size_t label = i % classes;
        labels[i] = label;
        for (size_t j = 0; j < features; j++) {
            double noise = gsl_ran_flat(rng, -0.6, 0.6);
            X->data[i * features + j] = centers[label][j] + noise;
        }
    }
    printf("------------\n");
    printIntArray(labels, batch);

    Linear* layer = createLinearLayer(features, classes);
    /* NOTE Not initializing the weights to make learning start from a random
     * guess */

    LearnableParam* params[] = {layer->weights, layer->bias};
    SGD optim = createSGD(0.05, params, 2);

    Tensor* logits = createTensor(batch, n, classes);
    Tensor* loss_grad = zerosLike(logits);

    CrossEntropyLoss ceLoss = createCrossEntropyLoss(classes);
    Tensor* loss = createVector(batch);

    size_t epochs = 10;
    double initial_loss = 0;
    double final_loss = 0;
    /* To perform inference on the training set */
    size_t decisions[batch];

    for (size_t i = 0; i < epochs; i++) {
        optim.zeroGrads(&optim);

        layer->forward(layer, X, logits);

        ceLoss.forward(&ceLoss, logits, labels, loss);

        double current_loss = gsl_stats_mean(loss->data, 1, loss->flat_size);
        final_loss = current_loss;
        if (i == 0) {
            initial_loss = current_loss;
        }

        simpleInference(batch, classes, logits, decisions);
        /* printf("Predictions:\n"); */
        /* printIntArray(decisions, batch); */
        printf("Loss: %.3f\n", current_loss);
        printf("Accuracy: %.2f\n", accuracyScore(decisions, labels, batch));

        /* Loss should be strictly decreasing in this example */
        TEST_ASSERT_LESS_OR_EQUAL_DOUBLE(final_loss, current_loss);

        ceLoss.backward(&ceLoss, labels, loss_grad);
        layer->backward(layer, loss_grad, X_grad);

        optim.step(&optim);
    }

    printf("Multi-Class Final Loss:   %.4f\n", final_loss);

    /* Random guess for 3 classes: -ln(1/3) = 1.0986 */
    TEST_ASSERT_DOUBLE_WITHIN(0.01, 1.098, initial_loss);
    TEST_ASSERT_LESS_THAN_DOUBLE(0.1, final_loss);

    layer->forward(layer, X, logits);
    simpleInference(batch, classes, logits, decisions);
    printf("Predictions:\n");
    printIntArray(decisions, batch);
    printf("Accuracy: %.2f\n", accuracyScore(decisions, labels, batch));

    destroyCrossEntropyLoss(&ceLoss);
    destroySGD(&optim);
    destroyLinearLayer(layer);
    destroyTensors(5, loss, loss_grad, X, X_grad, logits);
    destroyRNG(rng);
}

#endif  // TEST
