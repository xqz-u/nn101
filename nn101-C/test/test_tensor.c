#ifdef TEST

#include "unity.h"

#include <stddef.h>
#include <unistd.h>

#include "tensor.h"

#define SAMPLES 4
#define IN_DIMS 3
#define OUT_DIMS 2

#define IN_COUNT 3
#define IN_FEATURES 5
#define OUT_FEATURES 2
#define INPUT_FLAT_SIZE SAMPLES * IN_COUNT * IN_FEATURES
#define OUTPUT_FLAT_SIZE SAMPLES * IN_COUNT * OUT_FEATURES

static const double input_tensor_array[INPUT_FLAT_SIZE] = {3, 5, 5, 3, 2,
                                                           //
                                                           8, 8, 3, 8, 4,
                                                           //
                                                           7, 8, 9, 2, 4,
                                                           /*  */
                                                           2, 1, 8, 1, 5,
                                                           //
                                                           6, 4, 9, 9, 7,
                                                           //
                                                           8, 4, 9, 9, 9,
                                                           /*  */
                                                           5, 7, 6, 7, 9,
                                                           //
                                                           3, 2, 8, 2, 6,
                                                           //
                                                           2, 1, 4, 3, 1,
                                                           /*  */
                                                           7, 5, 5, 4, 1,
                                                           //
                                                           2, 5, 1, 3, 7,
                                                           //
                                                           1, 4, 7, 9, 2};

static Tensor* X = NULL;

void setUp(void) {
    X = createTensor(SAMPLES, IN_COUNT, IN_FEATURES);
    for (size_t i = 0; i < INPUT_FLAT_SIZE; i++) {
        X->data[i] = input_tensor_array[i];
    }
    /* printTensor(X); */
}

void tearDown(void) {
    destroyTensor(X);
    X = NULL;
    /* Wait a bit at the end of each test to allow `leaks` to properly record
     * memory usage */
    sleep(1);
}

void testEmptyVector(void) {
    double tmp[] = {0, 0, 0, 0, 0};
    Tensor* a = createVector(5);
    TEST_ASSERT_EQUAL_DOUBLE_ARRAY(tmp, a->data, a->features);
    destroyTensor(a);
}

void testDotProduct(void) {
    size_t size = 5;

    Tensor* a = createVector(size);
    for (size_t i = 0; i < a->features; i++) {
        a->data[i] = (double)(i + 1);
    }

    Tensor* b = createVector(size);
    for (size_t i = 0; i < b->features; i++) {
        b->data[i] = 1.f;
    };

    TEST_ASSERT_EQUAL_DOUBLE(15.f, dotProduct(a, b));

    destroyTensor(a);
    destroyTensor(b);
}

void test_matrixTranspose(void) {
    Tensor* M = createMatrix(SAMPLES, IN_FEATURES);

    for (size_t i = 0; i < SAMPLES; i++) {
        for (size_t j = 0; j < IN_FEATURES; j++) {
            size_t idx = indexMatrix(M, i, j);
            M->data[idx] = (double)(idx + 1);
        }
    }

    Tensor* Mt = createMatrix(M->features, M->n);
    transpose(M, Mt);

    /* printMatrix(Mt); */
    /* printf("\n"); */
    /* printMatrix(M); */
    /* printf("\n"); */

    double M_i[M->features];
    double Mt_i[Mt->n];
    for (size_t i = 0; i < SAMPLES; i++) {
        copyMatrixRow(i, M, M_i);
        copyMatrixCol(i, Mt, Mt_i);
        /* printRealArray(M_i, M->features); */
        /* printRealArray(Mt_i, Mt->n); */
        TEST_ASSERT_EQUAL_DOUBLE_ARRAY(M_i, Mt_i, Mt->features);
    }

    destroyTensor(M);
    destroyTensor(Mt);
}

void test_matrixProduct(void) {
    double x[SAMPLES * IN_DIMS] = {4, 2, 9,
                                   /*  */
                                   9, 4, 5,
                                   /*  */
                                   5, 5, 7,
                                   /*  */
                                   3, 3, 9};

    double w[IN_DIMS * OUT_DIMS] = {4, 5,
                                    /*  */
                                    2, 6,
                                    /*  */
                                    8, 3};

    double expected[SAMPLES * OUT_DIMS] = {92, 59,
                                           /*  */
                                           84, 84,
                                           /*  */
                                           86, 76,
                                           /*  */
                                           90, 60};

    Tensor A = {.data = x, .batch = 1, .n = SAMPLES, .features = IN_DIMS};
    Tensor B = {.data = w, .batch = 1, .n = IN_DIMS, .features = OUT_DIMS};

    /* printMatrix(A); */
    /* printf("\n"); */
    /* printMatrix(B); */
    /* printf("\n"); */

    Tensor* C = createMatrix(SAMPLES, OUT_DIMS);
    matMul(&A, &B, C);
    /* printMatrix(C); */

    for (size_t i = 0; i < C->n * C->features; i++) {
        TEST_ASSERT_EQUAL_DOUBLE(C->data[i], expected[i]);
    }

    destroyTensor(C);
}

void test_indexTensor(void) {
    size_t counter = 0;
    for (size_t k = 0; k < X->batch; k++) {
        for (size_t i = 0; i < X->n; i++) {
            for (size_t j = 0; j < X->features; j++) {
                double value = tensorGet(X, k, i, j);
                /* printf("(%d %d %d), %d -> %.2f, %.2f\n", k, i, j,
counter, */
                /*        value, input_tensor_array[counter]); */
                TEST_ASSERT_EQUAL_DOUBLE(value, input_tensor_array[counter]);
                counter++;
            }
        }
    }
}

/*
 * Simulate multiplying a batch of inputs (eg images) with a weight
 * matrix.
 * idea:
 * 1. create a random tensor SAMPLES x IN_DIMS x OUT_DIMS
 * 2. create a simple weight matrix
 * 3. for each matrix of the tensor, perform matrix multiplication
 * 4. fill the results in an output tensor
 * This should help inspecting the utility of the Tensor / MatrixView API
 * design.
 */
void test_batchedMatrixProduct(void) {
    double expected[OUTPUT_FLAT_SIZE] = {18, 36,
                                         //
                                         31, 62,
                                         //
                                         30, 60,
                                         /*  */
                                         17, 34,
                                         //
                                         35, 70,
                                         //
                                         39, 78,
                                         /*  */
                                         34, 68,
                                         //
                                         21, 42,
                                         //
                                         11, 22,
                                         /*  */
                                         22, 44,
                                         //
                                         18, 36,
                                         //
                                         23, 46};

    Tensor* ones = onesMatrix(IN_FEATURES, OUT_FEATURES);
    for (size_t i = 0; i < IN_FEATURES; i++) {
        size_t idx = indexMatrix(ones, i, 1);
        ones->data[idx] *= 2;
    }
    /* printMatrix(ones); */

    Tensor* Y = createTensor(SAMPLES, IN_COUNT, OUT_FEATURES);

    batchedMatMul(X, ones, Y);
    /* printTensor(Y); */

    for (size_t i = 0; i < OUTPUT_FLAT_SIZE; i++) {
        TEST_ASSERT_EQUAL_DOUBLE(Y->data[i], expected[i]);
    }

    destroyTensor(ones);
    destroyTensor(Y);
}

#endif  // TEST
