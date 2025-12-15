#include <arpa/inet.h>
#include <assert.h>
#include <gsl/gsl_statistics_double.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "mnist.h"

void read_mnist_header(FILE* fp,
                       const char* path,
                       size_t n_headers,
                       uint32_t* metadata) {
    if (fread(metadata, sizeof(uint32_t), n_headers, fp) != n_headers) {
        printf("Could not read MNIST header metatadata for file '%s'\n", path);
        return;
    }

    /* Data is stored in big-endian format in MNIST, make it little-endian
     * (basically all modern systems): *networklong* -> *hostlong* */
    for (size_t i = 0; i < n_headers; i++) {
        metadata[i] = ntohl(metadata[i]);
    }
}

uint8_t* read_mnist_labels(FILE* fp, const char* path, uint32_t n_samples) {
    size_t n_headers = 2;  // Magic number; number of labels
    uint32_t meta[n_headers] = {};
    read_mnist_header(fp, path, n_headers, meta);

    if (meta[0] != 2049) {
        printf("This might not be the correct labels file: '%s'\n", path);
        return nullptr;
    }

    assert(meta[1] == n_samples);

    uint8_t* labels = malloc(n_samples * sizeof(uint8_t));
    if (labels == nullptr) {
        return nullptr;
    }

    size_t elements_read = fread(labels, sizeof(uint8_t), meta[1], fp);
    if (elements_read != meta[1]) {
        printf("Read only %lu out of %d expected elements, file: '%s'\n",
               elements_read, meta[1], path);
        free(labels);
        return nullptr;
    }

    return labels;
}

bool read_mnist_metadata(FILE* fp,
                         const char* path,
                         MNISTMeta* metadata,
                         bool verbose) {
    /* Magic number; number of samples; sample width; sample height */
    size_t n_headers = 4;
    uint32_t meta[n_headers] = {};
    read_mnist_header(fp, path, n_headers, meta);

    if (meta[0] != 2051) {
        printf("This might not be the correct data file: '%s'\n", path);
        return false;
    }

    /* There's no padding between adjacent struct fields of the same type, so we
     * can copy into the struct directly. */
    memcpy(metadata, &meta[1], 3 * sizeof(uint32_t));
    metadata->flat_img_size = metadata->width * metadata->height;
    metadata->n_classes = 10;

    if (verbose) {
        printf("Magic number: %u\n", meta[0]);
    }

    return true;
}

bool read_mnist_data(FILE* fp,
                     const char* path,
                     MNISTMeta* meta,
                     uint8_t* images) {
    size_t dataset_pixels = meta->flat_img_size * meta->n_samples;

    size_t elements_read = fread(images, sizeof(uint8_t), dataset_pixels, fp);
    if (elements_read != dataset_pixels) {
        printf("Read only %lu out of %lu expected elements, file: '%s'\n",
               elements_read, dataset_pixels, path);
        return false;
    }
    return true;
}

void scale_uint8_array(size_t size, uint8_t* arr, double* scaled_arr) {
    for (size_t i = 0; i < size; i++) {
        scaled_arr[i] = (double)arr[i] / UINT8_MAX;
    }
}

void standardize_double_array(size_t size,
                              double* arr,
                              double summary_stats[2]) {
    summary_stats[0] = gsl_stats_mean(arr, 1, size);
    summary_stats[1] = gsl_stats_sd(arr, 1, size);

    for (size_t i = 0; i < size; i++) {
        arr[i] = (arr[i] - summary_stats[0]) / summary_stats[1];
    }
}

double* preprocess_mnist(size_t dataset_size,
                         uint8_t* raw_images,
                         double summary_stats[2]) {
    double* images = malloc(dataset_size * sizeof(double));
    if (images == nullptr) {
        return images;
    }

    scale_uint8_array(dataset_size, raw_images, images);
    standardize_double_array(dataset_size, images, summary_stats);

    return images;
}

double preproc_pixel(uint8_t v, double mean, double std) {
    double bv = (double)v / UINT8_MAX;
    return (bv - mean) / std;
}

void print_uint8_mnist_image(const uint8_t* images,
                             size_t idx,
                             uint32_t width,
                             uint32_t height) {
    uint8_t tresh_high = 200;
    uint8_t tresh_mid = 100;
    uint8_t tresh_low = 0;

    const uint8_t* image = images + (idx * width * height);

    for (size_t y = 0; y < height; y++) {
        size_t row_start = y * width;
        for (size_t x = 0; x < width; x++) {
            uint8_t pixel = image[row_start + x];
            if (pixel > tresh_high) {
                printf("##");
            } else if (pixel > tresh_mid) {
                printf("::");
            } else if (pixel > tresh_low) {
                printf("..");
            } else {
                printf("  ");
            }
        }
        printf("\n");
    }
}

void print_double_mnist_image(const double* images,
                              size_t idx,
                              uint32_t width,
                              uint32_t height,
                              double mean,
                              double std) {
    double tresh_high = preproc_pixel(200, mean, std);
    double tresh_mid = preproc_pixel(100, mean, std);
    double tresh_low = preproc_pixel(0, mean, std);

    const double* image = images + (idx * width * height);

    for (size_t y = 0; y < height; y++) {
        size_t row_start = y * width;
        for (size_t x = 0; x < width; x++) {
            double pixel = image[row_start + x];
            if (pixel > tresh_high) {
                printf("##");
            } else if (pixel > tresh_mid) {
                printf("::");
            } else if (pixel > tresh_low) {
                printf("..");
            } else {
                printf("  ");
            }
        }
        printf("\n");
    }
}

void print_dataset_info(const MNIST* dataset) {
    printf(
        "Number of images: %u\nImage width: %u\nImage "
        "height: %u\nFlat image size: %u\n Total samples: %u\n",
        dataset->meta->n_samples, dataset->meta->width, dataset->meta->height,
        dataset->meta->flat_img_size, dataset->meta->n_samples);

    printf("[0, 1] scaled dataset mean: %.5f std: %.5f\n", dataset->mean,
           dataset->std);

    for (size_t i = 0; i < 3; i++) {
        printf("Sample (reconstructed after preproc):\n");
        print_double_mnist_image(dataset->data, i, dataset->meta->width,
                                 dataset->meta->height, dataset->mean,
                                 dataset->std);
        printf("Label: %u\n", dataset->labels[i]);
    }
}

MNIST* create_mnist_dataset(DatasetPartition partition, bool verbose) {
    /* Declare all pointer variables here so we can use goto cleanup pattern &
     * avoid writing incremental free() blocks. */
    const char* data_path = nullptr;
    const char* labels_path = nullptr;
    FILE* data_fp = nullptr;
    FILE* labels_fp = nullptr;
    MNISTMeta* meta = nullptr;
    uint8_t* raw_images = nullptr;
    double* processed_images = nullptr;
    uint8_t* labels = nullptr;
    MNIST* dataset = nullptr;

    if (partition == TRAIN) {
        data_path = "/Users/xqz-u/datasets/mnist/train-images.idx3-ubyte";
        labels_path = "/Users/xqz-u/datasets/mnist/train-labels.idx1-ubyte";
    } else {
        data_path = "/Users/xqz-u/datasets/mnist/t10k-images.idx3-ubyte";
        labels_path = "/Users/xqz-u/datasets/mnist/t10k-labels.idx1-ubyte";
    }

    /* ------ Data ------ */
    if ((data_fp = fopen(data_path, "rb")) == nullptr) {
        printf("Error opening file '%s'\n", data_path);
        goto cleanup;
    }

    if ((meta = malloc(sizeof(MNISTMeta))) == nullptr) {
        printf("Error allocating pointer for MNIST metadata\n");
        goto cleanup;
    }

    if (!read_mnist_metadata(data_fp, data_path, meta, verbose)) {
        printf("Error opening file '%s'\n", data_path);
        goto cleanup;
    }

    size_t dataset_size = meta->n_samples * meta->flat_img_size;
    if ((raw_images = malloc(dataset_size * sizeof(uint8_t))) == nullptr) {
        printf("Error allocating uint8_t buffer for %lu images\n",
               dataset_size);
        goto cleanup;
    }

    if (!read_mnist_data(data_fp, data_path, meta, raw_images)) {
        printf("Error reading MNIST images at '%s'\n", data_path);
        goto cleanup;
    }

    double mean_std[2];
    if ((processed_images =
             preprocess_mnist(dataset_size, raw_images, mean_std)) == nullptr) {
        printf("Error processing raw MNIST images\n");
        goto cleanup;
    }

    fclose(data_fp);

    /* ------ Labels ------ */
    if ((labels_fp = fopen(labels_path, "rb")) == nullptr) {
        printf("Error opening file '%s'\n", labels_path);
        goto cleanup;
    }

    if ((labels = read_mnist_labels(labels_fp, labels_path, meta->n_samples)) ==
        nullptr) {
        printf("Error allocating uint8_t buffer for %u labels\n",
               meta->n_samples);
        goto cleanup;
    }

    fclose(labels_fp);

    if ((dataset = malloc(sizeof(MNIST))) == nullptr) {
        printf("Error allocating buffer for MNIST struct\n");
        goto cleanup;
    }
    dataset->partition = partition;
    dataset->meta = meta;
    dataset->data = processed_images;
    dataset->labels = labels;
    dataset->mean = mean_std[0];
    dataset->std = mean_std[1];

    if (verbose) {
        for (size_t i = 0; i < 3; i++) {
            printf("Sample (raw):\n");
            print_uint8_mnist_image(raw_images, i, dataset->meta->width,
                                    dataset->meta->height);
            printf("Label: %u\n", dataset->labels[i]);
        }
    }

    free(raw_images);
    return dataset;

cleanup:
    if (data_fp) {
        fclose(data_fp);
    }
    if (labels_fp) {
        fclose(labels_fp);
    }
    free(meta);
    free(raw_images);
    free(processed_images);
    free(labels);
    free(dataset);
    return dataset;
}

void destroy_mnist_dataset(MNIST* dataset) {
    if (dataset == nullptr) {
        return;
    }
    free(dataset->data);
    free(dataset->meta);
    free(dataset->labels);
    free(dataset);
}
