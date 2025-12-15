#include <gsl/gsl_randist.h>
#include <gsl/gsl_statistics_double.h>
#include <math.h>
#include <stdint.h>
#include <string.h>
#include <time.h>

#include "loss.h"
#include "mlp.h"
#include "mnist.h"
#include "sgd.h"
#include "tensor.h"
#include "utils.h"

typedef struct TrainState TrainState;
struct TrainState {
    size_t* sample_indices;
    size_t* targets;
    Tensor* X;
    Tensor* X_grad;
    Tensor* loss;
    size_t tot_batches;
    size_t dataset_size;
    gsl_rng* rng;
};

typedef struct HyperParams HyperParams;
struct HyperParams {
    size_t batch_size;
    size_t eval_batch_size;
    size_t train_epochs;
    size_t batch_print_interval;
    size_t eval_every;
    size_t n_hidden_dims;
    size_t* hidden_dims;
    double lr;
};

TrainState createTrainState(size_t batch_size, MNIST* ds) {
    size_t* indices = malloc(ds->meta->n_samples * sizeof(size_t));
    for (size_t i = 0; i < ds->meta->n_samples; i++) {
        indices[i] = i;
    }
    size_t* targets = malloc(batch_size * sizeof(size_t));

    Tensor* X = createTensor(batch_size, 1, ds->meta->flat_img_size);
    Tensor* X_grad = zerosLike(X);
    Tensor* loss = createVector(batch_size);
    gsl_rng* rng = initRNG();

    size_t dataset_size = ds->meta->n_samples;
    /* If dataset size is not perfectly divisible by the chosen batch size, make
     * sure to round up the number of batches to account for the last (smaller)
     * batch. */
    size_t tot_batches = (size_t)ceil((double)dataset_size / batch_size);

    return (TrainState){.sample_indices = indices,
                        .targets = targets,
                        .X = X,
                        .X_grad = X_grad,
                        .loss = loss,
                        .dataset_size = dataset_size,
                        .tot_batches = tot_batches,
                        .rng = rng};
}

void destroyTrainState(TrainState* ts) {
    if (ts == nullptr) {
        return;
    }
    free(ts->sample_indices);
    free(ts->targets);
    destroyRNG(ts->rng);
    destroyTensors(3, ts->X, ts->X_grad, ts->loss);
}

void printEpochProgress(size_t current_batch,
                        size_t print_interval,
                        size_t tot_batches,
                        double running_loss) {
    if (current_batch % print_interval == 0 ||
        current_batch == tot_batches - 1) {
        printf("\r\tBatch %lu/%lu | Loss: %.4f", current_batch, tot_batches,
               running_loss);
        fflush(stdout);
    }
}

/* NOTE a bunch of useless copies of data around, make it simpler & duplicate
 * some code */
double evaluate(MLP* model, MNIST* test_ds, size_t batch_size) {
    size_t tot_correct = 0;
    size_t dataset_size = test_ds->meta->n_samples;

    Tensor* X = createTensor(batch_size, 1, test_ds->meta->flat_img_size);

    for (size_t i = 0; i < dataset_size; i += batch_size) {
        size_t current_batch_size = batch_size;

        if (i + current_batch_size > dataset_size) {
            current_batch_size = dataset_size - i;

            X->batch = current_batch_size;
            X->flat_size = current_batch_size * X->features;
        }

        /* No need to shuffle the test dataset */
        memcpy(X->data, &test_ds->data[i * X->features],
               current_batch_size * X->features * sizeof(double));

        size_t casted_labels[current_batch_size];
        for (size_t j = 0; j < current_batch_size; j++) {
            casted_labels[j] = (size_t)test_ds->labels[i + j];
        }

        /* Inference & compute accuracy */
        model->forward(model, X);

        Tensor* logits = model->logits[model->n_layers - 1];

        size_t predictions[current_batch_size];
        simpleInference(current_batch_size, test_ds->meta->n_classes, logits,
                        predictions);

        tot_correct +=
            countCorrect(predictions, casted_labels, current_batch_size);
    }

    destroyTensor(X);

    return (double)tot_correct / dataset_size;
}

void ensureCompatibleBatchSize(TrainState* ts, size_t batch_size) {
    ts->X->batch = batch_size;
    ts->X->flat_size = batch_size * ts->X->features;

    ts->X_grad->batch = batch_size;
    ts->X_grad->flat_size = batch_size * ts->X->features;

    /* This is a vector, only the last dimension is used. */
    ts->loss->features = batch_size;
    ts->loss->flat_size = batch_size;
}

void gatherBatchData(size_t batch_size,
                     size_t start_idx,
                     TrainState* ts,
                     MNIST* ds) {
    size_t input_features = ts->X->features;
    /* Copy the batch data into a Tensor compatible for the model - and the
     * labels too. */
    for (size_t i = 0; i < batch_size; i++) {
        size_t sample_idx = ts->sample_indices[i + start_idx];
        memcpy(&ts->X->data[i * input_features],
               &ds->data[sample_idx * input_features],
               input_features * sizeof(double));
        ts->targets[i] = ds->labels[sample_idx];
    }
}

double trainOnBatch(TrainState* ts,
                    MLP* model,
                    SGD* optim,
                    CrossEntropyLoss* loss_fn) {
    size_t output_idx = model->n_layers - 1;

    optim->zeroGrads(optim);

    model->forward(model, ts->X);

    loss_fn->forward(loss_fn, model->logits[output_idx], ts->targets, ts->loss);

    /* TODO man I don't like this...*/
    ensureTensorCapacity(&model->logits_grad[output_idx],
                         model->logits[output_idx]);

    loss_fn->backward(loss_fn, ts->targets, model->logits_grad[output_idx]);

    /* TODO is it possible to avoid using X_grad? Intermediate grads are
     * only needed for the chain rule, which ends at the input layer... */
    model->backward(model, ts->X_grad);

    optim->step(optim);

    return gsl_stats_mean(ts->loss->data, 1, ts->loss->flat_size);
}

double trainOneEpoch(TrainState* ts,
                     MLP* model,
                     SGD* optim,
                     CrossEntropyLoss* loss_fn,
                     MNIST* train_ds,
                     HyperParams* hp) {
    /* Shuffle dataset idxs to get random training batches. */
    gsl_ran_shuffle(ts->rng, ts->sample_indices, ts->dataset_size,
                    sizeof(size_t));

    double epoch_loss = 0;

    size_t batch_n = 0;
    size_t batches_seen = 0;

    for (size_t i = 0; i < ts->dataset_size; i += hp->batch_size) {
        size_t current_batch_size = hp->batch_size;

        /* On the last epoch batch the size might be smaller so we update it
         * where needed. */
        if (i + current_batch_size > ts->dataset_size) {
            current_batch_size = ts->dataset_size - i;
            ensureCompatibleBatchSize(ts, current_batch_size);
        }

        gatherBatchData(current_batch_size, i, ts, train_ds);

        double batch_mean_loss = trainOnBatch(ts, model, optim, loss_fn);

        epoch_loss += (batch_mean_loss * current_batch_size);  // un-average)
        batches_seen += current_batch_size;

        printEpochProgress(batch_n, hp->batch_print_interval, ts->tot_batches,
                           epoch_loss / batches_seen);

        batch_n += 1;
    }

    /* Reset for next epoch. */
    ensureCompatibleBatchSize(ts, hp->batch_size);

    return epoch_loss / ts->dataset_size;
}

void trainOnMnist(HyperParams* hp) {
    MNIST* train_ds = create_mnist_dataset(TRAIN, false);
    MNIST* test_ds = create_mnist_dataset(EVAL, false);

    TrainState ts = createTrainState(hp->batch_size, train_ds);

    size_t n_features = hp->n_hidden_dims + 2;

    /* Create full MLP spec */
    size_t network_features[n_features];
    network_features[0] = (size_t)train_ds->meta->flat_img_size;
    for (size_t i = 0; i < hp->n_hidden_dims; i++) {
        network_features[i + 1] = hp->hidden_dims[i];
    }
    network_features[n_features - 1] = train_ds->meta->n_classes;

    MLP model = createMLP(n_features, network_features, ts.rng);

    CrossEntropyLoss loss = createCrossEntropyLoss(train_ds->meta->n_classes);

    SGD optim = createSGDFromModel(hp->lr, &model);

    for (size_t i = 0; i < hp->train_epochs; i++) {
        printf("Epoch %lu/%lu\n", i, hp->train_epochs);

        /* Train */
        clock_t t = clock();
        double train_loss =
            trainOneEpoch(&ts, &model, &optim, &loss, train_ds, hp);
        t = clock() - t;
        double time_elapsed = (double)t / CLOCKS_PER_SEC;
        printf("\n\tTrain time: %.2fs | Train Loss: %.4f", time_elapsed,
               train_loss);

        /* Test */
        if (i % hp->eval_every == 0 || i == hp->train_epochs - 1) {
            t = clock();
            double val_acc = evaluate(&model, test_ds, hp->eval_batch_size);
            t = clock() - t;
            time_elapsed = (double)t / CLOCKS_PER_SEC;
            printf("| Eval time: %.2fs | Val Acc: %.2f", time_elapsed, val_acc);
        }

        printf("\n");
    }

    destroy_mnist_dataset(train_ds);
    destroy_mnist_dataset(test_ds);
    destroyMLP(&model);
    destroySGD(&optim);
    destroyTrainState(&ts);
    destroyCrossEntropyLoss(&loss);
}

int main(void) {
    size_t hidden_dims[] = {128};
    size_t n_hidden_dims = sizeof(hidden_dims) / sizeof(hidden_dims[0]);

    HyperParams hp = {/* .train_epochs = 50, */
                      .train_epochs = 1,
                      .batch_size = 128,
                      .eval_batch_size = 256,
                      /* .batch_print_interval = 50, */
                      .batch_print_interval = 1,
                      .eval_every = 1,
                      .hidden_dims = hidden_dims,
                      .n_hidden_dims = n_hidden_dims,
                      .lr = 1e-2};
    trainOnMnist(&hp);
    return 0;
}
