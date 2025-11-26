#ifndef LOSS_H
#define LOSS_H

#include "tensor.h"

typedef struct MSELoss MSELoss;
struct MSELoss {
    void (*forward)(Tensor* pred, Tensor* target, Tensor* out);
    void (*backward)(Tensor* pred, Tensor* target, Tensor* out);
};

MSELoss createMSELoss(void);

typedef struct CrossEntropyLoss CrossEntropyLoss;
struct CrossEntropyLoss {
    size_t classes;
    Tensor* probabilities;

    void (*forward)(CrossEntropyLoss* self,
                    Tensor* logits,
                    size_t labels[],
                    Tensor* loss);

    void (*backward)(CrossEntropyLoss* self,
                     size_t labels[],
                     Tensor* grad_logits);
};

CrossEntropyLoss createCrossEntropyLoss(size_t classes);
void destroyCrossEntropyLoss(CrossEntropyLoss* loss);

#endif  // LOSS_H
