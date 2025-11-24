#ifndef LOSS_H
#define LOSS_H

#include "tensor.h"

typedef struct MSELoss MSELoss;
struct MSELoss {
    void (*forward)(Tensor* pred, Tensor* target, Tensor* out);
    void (*backward)(Tensor* pred, Tensor* target, Tensor* out);
};

MSELoss createMSE(void);

#endif  // LOSS_H
