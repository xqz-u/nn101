# nn101

## WIP

- [ ] Categorical loss module (cross entropy).
- [x] Regression test: check that a single linear layer is able to learn a
      simple regression task.
- [x] SGD.
- [ ] Numerical gradient tests (finite differences).
  - [x] Linear layer
- [x] Fix static graph computation being broken by inconsistent batch size during training.

## TODO

- [x] Configure `cloc`.
- [ ] Run tests on push/commit.
- [ ] Benchmark against BLAS implementation version.
- [ ] Benchmark against implementation using strides.
- [ ] ~~Benchmark dynamic vs. static computation graph allocation (?).~~ Does
      not seem super interesting...
- [ ] Document the whole process in some sort of essay.
- [ ] Compile with -O2 / -O3 + inspect assembly (-S) ?
- [ ] Logging ?
