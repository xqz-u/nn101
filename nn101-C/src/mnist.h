#ifndef MNIST_H
#define MNIST_H

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

typedef enum DatasetPartition { TRAIN, EVAL } DatasetPartition;

typedef struct MNISTMeta MNISTMeta;
struct MNISTMeta {
    uint32_t n_samples;
    uint32_t width;
    uint32_t height;
    uint32_t flat_img_size;
    size_t n_classes;
};

typedef struct MNIST MNIST;
struct MNIST {
    DatasetPartition partition;
    MNISTMeta* meta;
    double* data;
    uint8_t* labels;
    double mean;
    double std;
};

MNIST* create_mnist_dataset(DatasetPartition partition, bool verbose);
void destroy_mnist_dataset(MNIST* dataset);
void print_dataset_info(const MNIST* dataset);

#endif  // MNIST_H
