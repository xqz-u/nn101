#ifndef LEARNABLE_PARAM_H
#define LEARNABLE_PARAM_H

#include "tensor.h"

typedef struct LearnableParam LearnableParam;
struct LearnableParam {
    /* const char* name; */
    Tensor* param;
    Tensor* grad;
};

LearnableParam* createLearnableParam(size_t dim0, size_t dim1, size_t dim2);
void destroyLearnableParam(LearnableParam* self);

#endif  // LEARNABLE_PARAM_H
