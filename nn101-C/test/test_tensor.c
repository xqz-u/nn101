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

void test_matrixTranspose(void) {
    Matrix* M = createMatrix(SAMPLES, IN_FEATURES);

    for (size_t i = 0; i < SAMPLES; i++) {
        for (size_t j = 0; j < IN_FEATURES; j++) {
            size_t idx = indexMatrix(&M->view, i, j);
            M->view.data[idx] = (double)(idx + 1);
        }
    }

    Matrix* Mt = createMatrix(M->view.features, M->view.n);
    transpose(&M->view, &Mt->view);

    /* printMatrix(&Mt->view); */
    /* printf("\n"); */
    /* printMatrix(&M->view); */
    /* printf("\n"); */

    double M_i[M->view.features];
    double Mt_i[Mt->view.n];
    for (size_t i = 0; i < SAMPLES; i++) {
        copyMatrixRow(i, &M->view, M_i);
        copyMatrixCol(i, &Mt->view, Mt_i);
        /* printRealArray(M_i, M->view.features); */
        /* printRealArray(Mt_i, Mt->n); */
        TEST_ASSERT_EQUAL_DOUBLE_ARRAY(M_i, Mt_i, Mt->view.features);
    }

    destroyMatrix(M);
    destroyMatrix(Mt);
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

    MatrixView A = {.data = x, .n = SAMPLES, .features = IN_DIMS};
    MatrixView B = {.data = w, .n = IN_DIMS, .features = OUT_DIMS};

    /* printMatrix(&A->view); */
    /* printf("\n"); */
    /* printMatrix(&B->view); */
    /* printf("\n"); */

    Matrix* C = createMatrix(SAMPLES, OUT_DIMS);
    matMul(&A, &B, &C->view);
    /* printMatrix(&C->view); */

    for (size_t i = 0; i < C->view.n * C->view.features; i++) {
        TEST_ASSERT_EQUAL_DOUBLE(C->view.data[i], expected[i]);
    }

    destroyMatrix(C);
}

void test_indexTensor(void) {
    size_t counter = 0;
    for (size_t k = 0; k < X->batch; k++) {
        for (size_t i = 0; i < X->n; i++) {
            for (size_t j = 0; j < X->features; j++) {
                double value = tensorGet(X, k, i, j);
                /* printf("(%d %d %d), %d -> %.2f, %.2f\n", k, i, j, counter, */
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

    Matrix* ones = onesMatrix(IN_FEATURES, OUT_FEATURES);
    for (size_t i = 0; i < IN_FEATURES; i++) {
        double value = matrixGet(&ones->view, i, 1);
        matrixSet(&ones->view, i, 1, value * 2);
    }
    /* printMatrix(&ones->view); */

    Tensor* Y = createTensor(SAMPLES, IN_COUNT, OUT_FEATURES);

    batchedMatMul(X, &ones->view, Y);
    /* printTensor(Y); */

    for (size_t i = 0; i < OUTPUT_FLAT_SIZE; i++) {
        TEST_ASSERT_EQUAL_DOUBLE(Y->data[i], expected[i]);
    }

    destroyMatrix(ones);
    destroyTensor(Y);
}

#endif  // TEST
