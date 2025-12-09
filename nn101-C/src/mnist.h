#ifndef MNIST_H
#define MNIST_H

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

typedef struct MNISTMeta MNISTMeta;
typedef enum DatasetPartition { TRAIN, EVAL } DatasetPartition;
typedef struct MNIST MNIST;

MNIST* create_mnist_dataset(DatasetPartition partition, bool verbose);
void destroy_mnist_dataset(MNIST* dataset);
void print_dataset_info(const MNIST* dataset);

#endif  // MNIST_H
