#ifndef SGD_H
#define SGD_H

#include "nn_types.h"

typedef struct SGD SGD;
struct SGD {
    double lr;
    size_t params_count;
    LearnableParam* params;

    void (*step)(SGD optimizer);
};
SGD createSGD(double lr, LearnableParam params[], size_t params_count);
void destroySGD(SGD optimizer);

#endif  // SGD_H
