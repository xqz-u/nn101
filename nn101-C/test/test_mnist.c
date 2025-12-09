#ifdef TEST

#include "unity.h"

#include "mnist.h"
#include "tensor.h"
#include "utils.h"

void setUp(void) {}

void tearDown(void) {}

/* NOTE Not really a test, just driver code to check the datasets are
 * sensible... */
void test_mnistLoading(void) {
    MNIST* train_dataset = create_mnist_dataset(TRAIN, true);
    if (train_dataset == nullptr) {
        return;
    }

    print_dataset_info(train_dataset);

    destroy_mnist_dataset(train_dataset);

    MNIST* test_dataset = create_mnist_dataset(EVAL, true);
    if (test_dataset == nullptr) {
        return;
    }

    printf("\n");

    print_dataset_info(test_dataset);

    destroy_mnist_dataset(train_dataset);
}

#endif  // TEST
