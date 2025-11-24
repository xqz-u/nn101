#ifndef NN_TYPES_H
#define NN_TYPES_H

#include "tensor.h"

typedef struct LearnableParam LearnableParam;
struct LearnableParam {
    /* const char* name; */
    Tensor* param;
    Tensor* grad;
};

#endif  // NN_TYPES_H
