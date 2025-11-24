#ifdef TEST

#include "unity.h"

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

    LearnableParam params[] = {layer->weights, layer->bias};
    SGD optim = createSGD(0.01, params, 2);

    Tensor* Y_pred = zerosLike(Y);

    MSELoss mse = createMSE();
    Tensor* error = zerosLike(Y);
    Tensor* loss_grad = createTensor(X->batch, X->n, layer->out_features);
    Tensor* X_grad = zerosLike(X);

    double mean_err = -1.0;

    size_t epochs = 10;
    for (size_t i = 0; i < epochs; i++) {
        optim.zeroGrads(optim);

        layer->forward(layer, X, Y_pred);

        mse.forward(Y_pred, Y, error);

        double new_err = computeMean(error->data, error->flat_size);
        if (i == 0) {
            mean_err = new_err;
        }

        /* Loss should be strictly decreasing in this example */
        TEST_ASSERT_LESS_OR_EQUAL_DOUBLE(mean_err, new_err);

        mse.backward(Y_pred, Y, loss_grad);

        layer->backward(layer, loss_grad, X_grad);

        optim.step(optim);
        mean_err = new_err;
    }

    printf("%.5f\n", mean_err);

    for (size_t i = 0; i < 10; i++) {
        printf("%.2f ", Y->data[i]);
    }
    printf("\n");
    for (size_t i = 0; i < 10; i++) {
        printf("%.2f ", Y_pred->data[i]);
    }
    printf("\n");

    destroySGD(optim);
    destroyLinearLayer(layer);
    destroyTensor(error);
    destroyTensor(X_grad);
    destroyTensor(loss_grad);
    destroyTensor(X);
    destroyTensor(Y);
    destroyTensor(Y_pred);
    destroyRNG(rng);
}

#endif  // TEST
