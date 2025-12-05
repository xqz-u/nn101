#ifndef SGD_H
#define SGD_H

#include "learnable_param.h"

typedef struct SGD SGD;
struct SGD {
    double lr;
    size_t params_count;
    /* Array of LearnableParam* */
    LearnableParam** params;

    void (*step)(SGD* optimizer);
    void (*zeroGrads)(SGD* optimizer);
};

SGD createSGD(double lr, LearnableParam* params[], size_t params_count);
void destroySGD(SGD* optimizer);

#endif  // SGD_H
