#ifndef SGD_H
#define SGD_H

#include "learnable_param.h"
#include "mlp.h"

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
/* NOTE I don't like this, added to encapsulate params collection & not rewrite
 * tests... */
SGD createSGDFromModel(double lr, MLP* model);
void destroySGD(SGD* optimizer);

#endif  // SGD_H
