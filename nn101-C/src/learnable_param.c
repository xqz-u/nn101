#include "learnable_param.h"

LearnableParam* createLearnableParam(size_t dim0, size_t dim1, size_t dim2) {
    LearnableParam* p = malloc(sizeof(LearnableParam));

    p->param = createTensor(dim0, dim1, dim2);
    if (p->param == nullptr) {
        destroyLearnableParam(p);
        return nullptr;
    }

    p->grad = zerosLike(p->param);
    if (p->grad == nullptr) {
        destroyLearnableParam(p);
        return nullptr;
    }

    return p;
}

void destroyLearnableParam(LearnableParam* self) {
    if (self == nullptr) {
        return;
    }
    if (self->param != nullptr) {
        destroyTensor(self->param);
    }
    if (self->grad != nullptr) {
        destroyTensor(self->grad);
    }
    free(self);
}
