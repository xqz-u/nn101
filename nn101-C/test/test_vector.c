#ifdef TEST

#include "unity.h"

#include "vector.h"

void setUp(void) {}

void tearDown(void) {}

void testEmptyVector() {
    double tmp[] = {0, 0, 0, 0, 0};
    Vector* a = createVector(5);
    TEST_ASSERT_EQUAL_DOUBLE_ARRAY(tmp, a->data, a->size);
    destroyVector(a);
}

void testFilledVector() {
    double tmp[] = {1, 2, 3, 4, 5};
    Vector* a = createVector(5);
    fillVector(a, tmp);
    TEST_ASSERT_EQUAL_DOUBLE_ARRAY(tmp, a->data, a->size);
    destroyVector(a);
}

void testDotProduct() {
    size_t size = 5;

    Vector* a = createVector(size);
    for (size_t i = 0; i < size; i++) {
        a->data[i] = (double)(i + 1);
    }

    Vector* b = createVector(size);
    for (size_t i = 0; i < size; i++) {
        b->data[i] = 1.f;
    };

    TEST_ASSERT_EQUAL_DOUBLE(15.f, dotProduct(a, b));

    destroyVector(a);
    destroyVector(b);
}

#endif  // TEST
